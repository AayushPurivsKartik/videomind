# main.py  ← FINAL THEME-FIXED VERSION (Works on Gradio 5.x)
import gradio as gr
import os
import shutil
from utils.video_processor import extract_frames
from utils.vlm_captioner import captioner
from utils.searcher import VideoSearchEngine
import tempfile
engine = VideoSearchEngine()
current_video = None

def process_video(video):
    global current_video
    if video is None:
        return "Upload a video first!", None, gr.update(visible=False)

    # THE ULTIMATE FIX: Copy to our own safe location first
    if isinstance(video, str):  # Gradio 5.x sometimes passes path directly
        uploaded_path = video
    else:
        uploaded_path = video.name if hasattr(video, 'name') else video

    # Create a permanent copy in project folder
    dest_path = "uploaded_video.mp4"
    shutil.copy2(uploaded_path, dest_path)
    current_video = dest_path

    # Rest of your code — unchanged
    for folder in ["frames", "frames_detected", "embeddings"]:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

    status = "Extracting frames from video..."
    yield status, None, gr.update(visible=False)

    frames, timestamps, _ = extract_frames(dest_path, fps=1)

    status += f"\nExtracted {len(frames)} frames. Generating AI captions..."
    yield status, None, gr.update(visible=False)

    captions = []
    for i, frame in enumerate(frames):
        captions.append(captioner.describe(frame))
        if i % 5 == 0 or i == len(frames)-1:
            yield f"Captioning... {i+1}/{len(frames)} frames", None, gr.update(visible=False)

    status += f"\nBuilding CLIP + FAISS search engine..."
    yield status, None, gr.update(visible=False)

    engine.build(frames, captions, timestamps)

    return "VIDEO READY! Ask anything below", None, gr.update(visible=True)

def search_video(query):
    if current_video is None:
        return None, "Process a video first!"
    
    results = engine.search(query, top_k=9)
    
    images = []
    titles = []
    for r in results:
        from PIL import Image
        img = Image.open(r["image_path"])
        images.append(img)
        titles.append(f"{r['timestamp']:.1f}s | Score: {r['score']:.3f}\n{r['caption'][:120]}...")
    
    return images, f"Found {len(images)} moments for: \"{query}\""

# ────────────────────────── GRADIO UI (THEME REMOVED) ──────────────────────────
with gr.Blocks(title="VideoMind - Ask Anything About Your Video") as demo:  # ← No theme here - defaults to beautiful Gradio 5.x style
    gr.Markdown("# 🎥 VideoMind")
    gr.Markdown("### Upload any video → Ask natural language questions → Get exact moments with AI boxes & captions")
    gr.Markdown("Powered by **Florence-2 • YOLOv8 • CLIP • FAISS**")

    with gr.Row():
        video_input = gr.Video(label="Upload Video (.mp4, .mov, .avi)", height=300)
    
    process_btn = gr.Button("Process Video & Build AI Brain", variant="primary", size="lg")
    status_box = gr.Textbox(label="Status", lines=4)
    
    with gr.Row():
        query = gr.Textbox(label="Ask Anything", placeholder="e.g. Show me when a car is turning left", scale=4)
        search_btn = gr.Button("Search", variant="secondary", size="lg", visible=False)
    
    gr.Markdown("### Results")
    gallery = gr.Gallery(label="Matching Moments", columns=3, height="auto", object_fit="contain")
    result_text = gr.Textbox(label="Summary")

    process_btn.click(
        process_video,
        inputs=video_input,
        outputs=[status_box, gallery, search_btn]
    )
    
    search_btn.click(
        search_video,
        inputs=query,
        outputs=[gallery, result_text]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)