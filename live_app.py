import gradio as gr
import model_functions as mf

def run_calibration_sequence(): 
    yield "Cleaning data...", "", "", "", None 
    
    df_clean, S, status1 = mf.getting_data()
    yield status1, "Calculating Yield Curve...", "Waiting...", "", None
    
    
    df_clean, status2 = mf.div_yield(df_clean, S)
    yield status1, status2, "Calibrating Model (This may take time)...", "", None

    q = mf.get_live_yield()  
    df_clean, status3 = mf.get_model_prices(df_clean, S, q)
    yield status1, status2, status3, "Calculating Metrics & Plotting...", None
    
    fig = mf.compare_prices(df_clean)
    metrics = mf.evaluate_model(df_clean)
    yield status1, status2, status3, metrics, fig


with gr.Blocks(title="Merton's Jump-Diffusion Pricing Engine") as demo:
    gr.Markdown("# 📉 S&P 500 Merton's Jump-Diffusion  Pricing Engine")
    gr.Markdown("""
    > **⚠️ Note on Data Accuracy:** This application fetches free data via `yfinance`. During live market hours, options data is often **delayed by ~15 minutes**, while the underlying index price might be real-time. 
    > This synchronization mismatch can temporarily increase model error (MAPE/R^2). For the most precise calibration, results are best viewed **after market close**.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            
            inp_q = gr.Number(label="Dividend Yield (q)", value=0.015, visible=False)
            
            btn_calib = gr.Button("1. Run Calibration (Live Data)", variant="primary")
            
            
            clean_status = gr.Textbox(label="Data Cleaning Status", interactive=False)
            yield_status = gr.Textbox(label="Yield Curve Status", interactive=False)
            calib_status = gr.Textbox(label="Calibration Status", interactive=False)
            txt_metrics = gr.Textbox(label="Model Accuracy Metrics", lines=5)
            
            gr.Markdown("### 2. Price a Specific Option")
            inp_strike = gr.Number(label="Strike Price", value=4600)
            inp_mat = gr.Number(label="Maturity (Years)", value=1.0)
            btn_price = gr.Button("Calculate Price")
            
        with gr.Column(scale=1):
            out_price = gr.Textbox(label="Estimated Option Price & Details", lines=5, interactive=False)
            plot_3d = gr.Plot(label="Calibration Surface")
            
    # Actions
    btn_calib.click(
        fn=run_calibration_sequence, 
        inputs=[], 
        outputs=[clean_status, yield_status, calib_status, txt_metrics, plot_3d]
    )
    
    btn_price.click(
        fn=mf.price_arbitrary_option,
        inputs=[inp_strike, inp_mat],
        outputs=[out_price]
    )

if __name__ == "__main__":
    demo.queue().launch()