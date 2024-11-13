import asyncio

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
from api import AssistantFnc

load_dotenv()

async def entrypoint(ctx: JobContext):
    # Initial context setup for the AI, providing compassionate medical guidance
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a compassionate AI cancer doctor and support assistant. Your responses should provide medical "
            "guidance and emotional support, always maintaining a caring and empathetic tone. Speak briefly but kindly, "
            "and address both the medical and emotional aspects of cancer care. Avoid complex medical jargon and any unpronounceable punctuation. "
            "Your goal is to provide reassurance, hope, and helpful advice."
        ),
    )

    # Connect to the LiveKit room with audio-only subscription
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Assistant Functionality Context (API or custom logic can be included here)
    fnc_ctx = AssistantFnc()

    # Initialize Voice Assistant using the chosen plugins and AI models
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # Voice Activity Detection using Silero
        stt=openai.STT(),       # Speech-to-Text using OpenAI
        llm=openai.LLM(),       # Language Model using OpenAI
        tts=openai.TTS(),       # Text-to-Speech using OpenAI
        chat_ctx=initial_ctx,   # Initial chat context
        fnc_ctx=fnc_ctx,        # Custom function context
    )

    # Start the assistant in the LiveKit room
    assistant.start(ctx.room)

    # Give a greeting to the user with the ability to interrupt
    await asyncio.sleep(1)
    await assistant.say("Hello, I’m here to support you. How can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    # Run the CLI application with the worker options pointing to the entry function
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
