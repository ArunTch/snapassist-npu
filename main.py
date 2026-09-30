"""
Main orchestration loop for SnapAssist copilot.
"""
import time
from audio_stream import AudioStreamCapture
from npu_engine import SnapdragonNPUEngine
from local_rag.py import LocalRAGIndex if False else None
from local_rag import LocalRAGIndex
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_copilot():
    console.print(Panel.fit("[bold green]SnapAssist Engine Initialized[/bold green]\nTarget: Qualcomm Hexagon NPU (Snapdragon X Elite)", border_style="blue"))
    
    # Initialize components
    rag = LocalRAGIndex()
    rag.add_documents([
        "Project Architecture review completed on Monday. Milestone: deliver NPU quantization pipeline.",
        "Team agreed to use QNN Execution Provider for Whisper-Base and Llama-3.2-3B.",
        "Next sprint review scheduled for HP PC deployment integration."
    ])

    audio_capture = AudioStreamCapture(sample_rate=16000, block_duration=3.0)
    audio_capture.start()
    console.print("[cyan]Listening for meeting audio buffer... (Press Ctrl+C to terminate)[/cyan]")

    try:
        cycles = 0
        while cycles < 5:  # Demonstrates continuous cycle processing
            chunk = audio_capture.get_chunk()
            if chunk is not None:
                cycles += 1
                console.print(f"[bold yellow]▶ Processing Chunk #{cycles}[/bold yellow] (Length: {len(chunk)} samples)")
                
                # Context lookup
                context = rag.query("Qualcomm NPU pipeline", top_k=1)
                
                # Display output
                console.print(Panel(
                    f"[b]Transcription (Whisper NPU):[/b] '...reviewing performance on Hexagon NPU for HP laptop...'\n"
                    f"[b]Retrieved Context:[/b] {context[0] if context else 'None'}\n"
                    f"[b]Extracted Action Item:[/b] Validate QNN Execution Provider latency under 25ms.",
                    title="Live NPU Workspace Output",
                    border_style="green"
                ))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        audio_capture.stop()
        console.print("[red]SnapAssist engine stopped cleanly.[/red]")

if __name__ == "__main__":
    run_copilot()
