from django.conf import settings
from django.http import JsonResponse
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from langchain_core.messages import HumanMessage
import openai

openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

interpreter_assistant = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
    client=openai_client
)

def assistant_math_view(request):
    user_input = request.GET.get('user_input', '')

    if not user_input:
        return JsonResponse({"error": "User input is required"}, status=400)

    try:
        response_message = interpreter_assistant.invoke(HumanMessage(content=user_input))
        
        response_text = response_message.content
        
        return JsonResponse({"response": response_text})
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
