import requests

response = requests.post(
    "http://aita-pipeline-elb-1613440940.us-east-1.elb.amazonaws.com/pro",
    json={
        "id": "-OSKEZuUFKZ-J-7TEjaW",
        "filename": "20250514_MoMAH HC Studies_Kick-off_vForTranslation",
        "pdfUrl": "https://firebasestorage.googleapis.com/v0/b/snb-ai-translation-agent.firebasestorage.app/o/uploads%2F1749479238520_20250514_MoMAH%20HC%20Studies_Kick-off_vForTranslation.pdf?alt=media&token=38031a96-87c5-459e-b2ad-d36ad5048bd9",
        "pptUrl": "https://firebasestorage.googleapis.com/v0/b/snb-ai-translation-agent.firebasestorage.app/o/uploads%2F1749479210638_20250514_MoMAH%20HC%20Studies_Kick-off_vForTranslation.pptx?alt=media&token=d5ae6eca-3113-4e0b-b9bc-0fc53d0bc416"
    }
)

print(response.status_code)
print(response.text)