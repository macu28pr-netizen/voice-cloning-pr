import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "TTS==0.22.0",
        "gradio==4.44.0",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "transformers==4.36.2",
    )
    .env({"COQUI_TOS_AGREED": "1"})
)

app = modal.App("voice-cloning-server")


@app.function(
    gpu=modal.gpu.T4(),
    timeout=600,
    container_idle_timeout=120,
)
@modal.web_server(7860, startup_timeout=300)
def run_gradio():
    import gradio as gr
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def clonar_voz(texto, audio_referencia, idioma):
        if not texto or audio_referencia is None:
            return None

        archivo_salida = "/tmp/voz_clonada.wav"
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_referencia,
            language=idioma,
            file_path=archivo_salida,
        )
        return archivo_salida

    interfaz = gr.Interface(
        fn=clonar_voz,
        inputs=[
            gr.Textbox(
                label="Texto a convertir en voz",
                placeholder="Escribe aquí...",
            ),
            gr.Audio(type="filepath", label="Muestra de voz limpia"),
            gr.Dropdown(choices=["es", "en"], value="es", label="Idioma"),
        ],
        outputs=gr.Audio(label="Voz Clonada"),
        title="Clonación de Voz en Modal",
        description="Sube un archivo de voz y escribe el texto.",
    )

    gr.close_all()
    interfaz.queue().launch(server_name="0.0.0.0", server_port=7860)
