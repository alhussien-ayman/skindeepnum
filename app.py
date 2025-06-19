# app.py
import os
import json
from flask import Flask, render_template, request
from dotenv import load_dotenv
import plotly
import plotly.graph_objects as go
import numpy as np
from flask import Flask
import constants as C
import explicit_finite_difference as explicit
import implicit
import crank_nicolson
import finite_element_method as FEM
import method_of_lines as MOL
import analytical
from error_and_time import compute_relative_error  



load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('GEMINI_API_KEY', 'a-default-secret-key-for-development')

sim_functions = {
    'fd_explicit': explicit.solve_finite_difference_explicit,
    'fd_implicit': implicit.solve_finite_difference_implicit,
    'fd_crank_nicolson': crank_nicolson.solve_crank_nicolson_newton,
    'mol': MOL.solve_method_of_lines,
    'fem': FEM.solve_finite_element_method,
    'analytical': analytical.solve_analytical,
}

method_labels = {
    'fd_explicit': 'Explicit FD',
    'fd_implicit': 'Implicit FD',
    'fd_crank_nicolson': 'Crank-Nicolson (Newton)',
    'mol': 'Method of Lines',
    'fem': 'Finite Element Method',
    'analytical': 'Analytical Solution',
}

def create_plot(data, time_labels, title):
    if not data or not time_labels: return None
    fig = go.Figure()
    for i, label in enumerate(time_labels):
        if label not in data[0]: continue
        fig.add_trace(go.Scatter(
            x=[item['r'] for item in data],
            y=[item.get(label) for item in data],
            mode='lines', name=label,
            line=dict(color=C.LINE_COLORS[i % len(C.LINE_COLORS)])
        ))
    fig.update_layout(
        title=title, xaxis_title='Position r (cm)', yaxis_title='u(r,t) density',
        legend_title='Time', template='plotly_white', margin=dict(l=40, r=40, t=80, b=40),
        yaxis=dict(range=[0, 1.05])
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_relative_error_plot(params, method1, method2):
    result = compute_relative_error(params, method1, method2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result['time_labels'],
        y=result['relative_errors'],
        mode='lines+markers',
        name=f"{result['label2']} vs {result['label1']}",
        line=dict(color='crimson')
    ))
    fig.update_layout(
        title=f"Relative Error: {result['label2']} vs {result['label1']}",
        xaxis_title="Time Label",
        yaxis_title="Relative L2 Error",
        legend_title="Comparison",
        template='plotly_white',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        "parameter_configs": C.PARAMETER_CONFIGS,
        "params": C.DEFAULT_SIMULATION_PARAMS,
        "errors": {},
        "plots": {},
        "display_method": "all",
        "performance_results_text": "",
        "method_labels": method_labels 
    }

    if request.method == 'POST':
        params, errors = {}, {}

        for p_config in C.PARAMETER_CONFIGS:
            key = p_config['id']
            value_str = request.form.get(key, '').strip()
            try:
                val = float(value_str)
                if (p_config.get('min') is not None and val < p_config['min']) or \
                   (p_config.get('max') is not None and val > p_config['max']):
                    raise ValueError("Out of bounds")
                params[key] = val
            except (ValueError, TypeError):
                errors[key] = f"Invalid value. Must be a number within [{p_config.get('min', 'N/A')}, {p_config.get('max', 'N/A')}]."
                params[key] = p_config['defaultValue']

        context.update({
            "params": params,
            "errors": errors,
            "display_method": request.form.get('displayMethod', 'all')
        })

        if errors:
            return render_template('index.html', **context)

        # Run all methods
        all_results = {name: func(params) for name, func in sim_functions.items()}

        # Performance summary
        perf_lines = [format_perf_line(res['result']) for res in all_results.values()]
        header = f"{'Method':<28} | {'Grid Points':>11} | {'Time Steps':>10} | {'CPU Time (s)':>12}"
        context['performance_results_text'] = "\n".join([header, "-" * len(header)] + perf_lines)

        plots = {}
        for name, res in all_results.items():
            plot_data = transform_to_plot_data(res['result']['r'], res['result']['solutions'], res['tOutputLabels'])
            plots[name] = create_plot(plot_data, res['tOutputLabels'], res['result']['methodName'])

        # Comparison at final time
        comp_data, comp_labels = generate_comparison_data(all_results)
        plots['comparison'] = create_plot(comp_data, comp_labels, f"Comparison at t = {params['days']} days")

        # === Relative Error Bar Chart vs MOL ===
         
        ref_key = 'mol'
        ref_result = all_results.get(ref_key, {}).get('result', None)

        if ref_result:
            error_bars = []
            error_over_time = []

            for key, method_data in all_results.items():
                if key == ref_key:
                    continue
                curr_result = method_data['result']

                try:
                    error_result = compute_relative_error(params, sim_functions[ref_key], sim_functions[key])
                    avg_error = np.mean(error_result['relative_errors'])
                    error_bars.append({
                        'method': curr_result['methodName'],
                        'relative_error': avg_error
                    })

                    error_over_time.append(go.Scatter(
                        x=error_result['time_labels'],
                        y=error_result['relative_errors'],
                        mode='lines+markers',
                        name=f"{curr_result['methodName']} vs Method of Lines"
                    ))

                except Exception:
                    error_bars.append({
                        'method': curr_result['methodName'],
                        'relative_error': np.nan
                    })

            # Bar Chart
            error_bar_fig = go.Figure()
            error_bar_fig.add_trace(go.Bar(
                x=[e['method'] for e in error_bars],
                y=[e['relative_error'] for e in error_bars],
                marker_color='firebrick'
            ))
            error_bar_fig.update_layout(
                title='Average Relative L2 Error vs Method of Lines',
                xaxis_title='Method',
                yaxis_title='Average Relative Error',
                template='plotly_white',
                margin=dict(l=40, r=40, t=60, b=40)
            )
            plots['relative_error'] = json.dumps(error_bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Line Plot
            error_time_fig = go.Figure(data=error_over_time)
            error_time_fig.update_layout(
                title='Relative Error Over Time vs Method of Lines',
                xaxis_title='Time (days)',
                yaxis_title='Relative Error',
                template='plotly_white',
                margin=dict(l=40, r=40, t=60, b=40)
            )
            plots['relative_error_over_time'] = json.dumps(error_time_fig, cls=plotly.utils.PlotlyJSONEncoder)

        context['plots'] = plots

    return render_template('index.html', **context)



@app.route('/compare', methods=['GET', 'POST'])
@app.route('/compare', methods=['GET', 'POST'])
@app.route('/compare', methods=['GET', 'POST'])
def compare_methods_ui():
    context = {
        "sim_functions": sim_functions.keys(),
        "method_labels": method_labels,
        "parameter_configs": C.PARAMETER_CONFIGS,
        "params": C.DEFAULT_SIMULATION_PARAMS,
        "errors": {},
        "plots": {}
    }

    if request.method == 'POST':
        try:
            params, errors = {}, {}
            for config in C.PARAMETER_CONFIGS:
                key = config['id']
                value_str = request.form.get(key, '').strip()
                try:
                    val = float(value_str)
                    if (config.get('min') is not None and val < config['min']) or \
                       (config.get('max') is not None and val > config['max']):
                        raise ValueError("Out of bounds")
                    params[key] = val
                except (ValueError, TypeError):
                    errors[key] = f"Invalid value. Must be within [{config.get('min', '-')}, {config.get('max', '-')}]"
                    params[key] = config['defaultValue']

            selected_method1 = request.form['method1']
            selected_method2 = request.form['method2']

            method1 = sim_functions[selected_method1]
            method2 = sim_functions[selected_method2]

            if errors:
                context.update({"params": params, "errors": errors, "selected_method1": selected_method1,
                                "selected_method2": selected_method2})
                return render_template("compare.html", **context)

            result = compute_relative_error(params, method1, method2)

            # Relative Error Plot
            error_trace = go.Scatter(
                x=result['time_labels'],
                y=result['relative_errors'],
                mode='lines+markers',
                name=f"{result['label2']} vs {result['label1']}"
            )
            error_fig = go.Figure(data=[error_trace])
            error_fig.update_layout(title='Relative Error Over Time',
                                    xaxis_title='Time (days)', yaxis_title='Relative Error')
            error_plot_json = error_fig.to_json()

            # Time Comparison Plot
            time_trace = go.Bar(
                x=[result['label1'], result['label2']],
                y=[result['time_1'], result['time_2']],
                marker_color=['steelblue', 'darkorange']
            )
            time_fig = go.Figure(data=[time_trace])
            time_fig.update_layout(title='CPU Time Comparison',
                                   xaxis_title='Method', yaxis_title='CPU Time (s)')
            time_plot_json = time_fig.to_json()

            context.update({
                "plots": {
                    "relative_error": error_plot_json,
                    "time_comparison": time_plot_json
                },
                "params": params,
                "errors": errors,
                "selected_method1": selected_method1,
                "selected_method2": selected_method2
            })

        except Exception as e:
            context["error"] = str(e)

    return render_template("compare.html", **context)


       
def transform_to_plot_data(r_coords, solutions, time_labels):
    if not all([r_coords, time_labels]): return []
    data = []
    for r_idx, r_val in enumerate(r_coords):
        point = {'r': round(r_val, 4)}
        for t_idx, u_profile in enumerate(solutions):
            if t_idx < len(time_labels) and r_idx < len(u_profile):
                point[time_labels[t_idx]] = round(u_profile[r_idx], 4) if u_profile[r_idx] is not None else None
        data.append(point)
    return data

def generate_comparison_data(all_results):
    ref_res = next((res for res in all_results.values() if res['result']['r']), None)
    if not ref_res: return [], []
    r_coords = ref_res['result']['r']

    comp_data, comp_labels = [], []
    for name, res in all_results.items():
        if res['result']['solutions'] and not np.isnan(res['result']['solutions'][-1]).all():
            comp_labels.append(res['result']['methodName'])

    for r_idx, r_val in enumerate(r_coords):
        point = {'r': round(r_val, 4)}
        for name, res in all_results.items():
            if res['result']['methodName'] in comp_labels:
                final_profile = res['result']['solutions'][-1]
                if r_idx < len(final_profile):
                    point[res['result']['methodName']] = round(final_profile[r_idx], 4)
        comp_data.append(point)
    return comp_data, comp_labels

def format_perf_line(res):
    name = res['methodName'].ljust(28)
    grid = str(res.get('gridPoints', 'N/A')).rjust(11)
    steps = str(res.get('timeSteps', 'N/A')).rjust(10)
    cpu = f"{res.get('time', 0):.3f}".rjust(12)
    return f"{name} | {grid} | {steps} | {cpu}"


def create_app():
   return app