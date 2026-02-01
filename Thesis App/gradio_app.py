
"""Main Gradio application file."""

import gradio as gr
import logging

# Import configuration and utilities
from app_config import setup_logging, UI_CONFIG, CLASS_OPTIONS, DATAFRAME_COLUMNS
from ui_components import (
    select_all_classes, 
    select_common_classes, 
    clear_selection, 
    get_example_files
)
from app_logic import process_image, handle_export
from inference import initialize_model, inference_gradio


def create_gradio_interface():
    """Create and configure the Gradio interface.
    
    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(title=UI_CONFIG['title'], theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# 🛣️ {UI_CONFIG['title']}")
        gr.Markdown(UI_CONFIG['description'])
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                input_image = gr.Image(
                    label="Input Image", 
                    type="pil", 
                    height=UI_CONFIG['input_image_height']
                )
                
                with gr.Row():
                    confidence = gr.Slider(
                        minimum=UI_CONFIG['confidence_min'], 
                        maximum=UI_CONFIG['confidence_max'], 
                        value=UI_CONFIG['confidence_default'], 
                        step=UI_CONFIG['confidence_step'], 
                        label="Confidence Threshold"
                    )
                    submit_btn = gr.Button("🔍 Detect Objects", variant="primary")
                
                # Class selection section
                gr.Markdown("### 🎯 Class Selection")
                class_selector = gr.CheckboxGroup(
                    choices=CLASS_OPTIONS,
                    value=CLASS_OPTIONS,  # All selected by default
                    label="Select Classes to Detect",
                    info="Choose which arrow types you want to detect"
                )
                
                # Quick selection buttons
                with gr.Row():
                    select_all_btn = gr.Button("✅ Select All", size="sm")
                    select_common_btn = gr.Button("⭐ Common Classes", size="sm")
                    clear_all_btn = gr.Button("❌ Clear All", size="sm")
            
            with gr.Column(scale=1):
                # Export section
                gr.Markdown("### 💾 Export Results")
                
                with gr.Row():
                    export_format = gr.Radio(
                        choices=["JSON", "CSV"], 
                        value="JSON", 
                        label="Export Format"
                    )
                
                with gr.Row():
                    export_json_btn = gr.Button("📄 Export as JSON", size="sm")
                    export_csv_btn = gr.Button("📊 Export as CSV", size="sm")
                
                export_file = gr.File(label="📥 Download File", visible=True)
                export_status = gr.Textbox(
                    label="Export Status", 
                    lines=UI_CONFIG['export_status_lines'], 
                    placeholder="Export status will appear here..."
                )
        
        # Results section
        with gr.Row():
            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="Detection Results", 
                    type="pil", 
                    height=UI_CONFIG['output_image_height']
                )
                
            with gr.Column(scale=1):
                # Detection results dataframe
                detection_dataframe = gr.Dataframe(
                    label="📊 Detection Results",
                    headers=DATAFRAME_COLUMNS['headers'],
                    datatype=DATAFRAME_COLUMNS['datatypes'],
                    max_rows=UI_CONFIG['max_dataframe_rows'],
                    overflow_row_behaviour="paginate"
                )
                
                # Summary text below the dataframe
                summary_text = gr.Textbox(
                    label="Summary", 
                    lines=UI_CONFIG['summary_lines'],
                    interactive=False,
                    placeholder="Detection summary will appear here..."
                )
        
        # Set up event handlers
        _setup_event_handlers(
            demo, input_image, confidence, class_selector, submit_btn,
            select_all_btn, select_common_btn, clear_all_btn,
            export_json_btn, export_csv_btn, export_file, export_status,
            output_image, detection_dataframe, summary_text
        )
        
        # Add examples if available
        example_files = get_example_files()
        if example_files:
            gr.Examples(
                examples=example_files,
                inputs=input_image,
                label="🖼️ Example Images"
            )
    
    return demo


def _setup_event_handlers(
    demo, input_image, confidence, class_selector, submit_btn,
    select_all_btn, select_common_btn, clear_all_btn,
    export_json_btn, export_csv_btn, export_file, export_status,
    output_image, detection_dataframe, summary_text
):
    """Set up all event handlers for the Gradio interface."""
    
    # Main detection action
    submit_btn.click(
        fn=lambda img, thresh, classes: process_image(
            img, thresh, classes, global_model, inference_gradio
        ),
        inputs=[input_image, confidence, class_selector],
        outputs=[output_image, detection_dataframe, summary_text]
    )
    
    # Quick selection button actions
    select_all_btn.click(
        fn=select_all_classes,
        inputs=[],
        outputs=[class_selector]
    )
    
    select_common_btn.click(
        fn=select_common_classes,
        inputs=[],
        outputs=[class_selector]
    )
    
    clear_all_btn.click(
        fn=clear_selection,
        inputs=[],
        outputs=[class_selector]
    )
    
    # Export actions
    export_json_btn.click(
        fn=lambda: handle_export("json"),
        inputs=[],
        outputs=[export_file, export_status]
    )
    
    export_csv_btn.click(
        fn=lambda: handle_export("csv"),
        inputs=[],
        outputs=[export_file, export_status]
    )


def main():
    """Main function to initialize and launch the Gradio app."""
    # Setup logging
    setup_logging()
    
    # Initialize the model on startup
    logging.info("Initializing model...")
    try:
        global global_model
        global_model = initialize_model()
        logging.info("Model initialized successfully")
    except Exception as e:
        logging.exception("Error initializing model")
        raise e
    
    # Create and launch the interface
    logging.info("Creating UI...")
    demo = create_gradio_interface()
    logging.info("UI Created")
    
    logging.info("Launching demo...")
    demo.launch(share=True)
    logging.info("Demo launched")


if __name__ == "__main__":
    main()
