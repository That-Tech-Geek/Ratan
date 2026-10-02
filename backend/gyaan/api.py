from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET,require_POST
import json
QUESTIONS=[{"id":"M8-001","class":8,"subject":"maths","prompt":"What is 3/4 + 1/4?","options":["1","3/8","4/8","2"],"answer":"1"}]
@require_GET
def health(request):return JsonResponse({"service":"gyaan-saathi-api","status":"ok"})
@require_GET
def questions(request):
 cls=int(request.GET.get("class","8"));subject=request.GET.get("subject","maths")
 return JsonResponse({"version":"1","class":cls,"subject":subject,"questions":[q for q in QUESTIONS if q["class"]==cls and q["subject"]==subject]})
@csrf_exempt
@require_POST
def sync_batch(request):
 try:body=json.loads(request.body);items=body.get("items",[])
 except json.JSONDecodeError:return JsonResponse({"error":"invalid_json"},status=400)
 if not isinstance(items,list) or len(items)>20:return JsonResponse({"error":"batch_limit"},status=400)
 return JsonResponse({"accepted":len(items),"rejected":0,"idempotency_ids":[x.get("id") for x in items]})