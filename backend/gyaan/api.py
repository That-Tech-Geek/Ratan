from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET,require_POST
from django.utils.dateparse import parse_datetime
from .models import AuditLog,DiagnosticSession,DiagnosticResponse,Student
import json

QUESTIONS=[{"id":"M8-001","class":8,"subject":"maths","prompt":"What is 3/4 + 1/4?","options":["1","3/8","4/8","2"],"answer":"1"}]
LIKERT=[{"id":f"L-{i:02d}","text":text} for i,text in enumerate([
"Pictures help me understand new ideas.","I learn well when someone explains aloud.",
"I remember what I write down.","I like learning by doing.","I prefer examples before rules.",
"I can explain a lesson after hearing it.","I like reading short explanations.","I learn from practice questions.",
"I like diagrams and charts.","Talking through a problem helps me.","Writing steps helps me remember.","Hands-on activities help me learn."
],1)]

@require_GET
def health(request): return JsonResponse({"service":"gyaan-saathi-api","status":"ok"})

@require_GET
def questions(request):
    cls=int(request.GET.get("class","8")); subject=request.GET.get("subject","maths")
    return JsonResponse({"version":"1","class":cls,"subject":subject,"questions":[q for q in QUESTIONS if q["class"]==cls and q["subject"]==subject]})

@require_GET
def likert_items(request):
    return JsonResponse({"version":"1","language":request.GET.get("language","en"),"items":LIKERT})

@csrf_exempt
@require_POST
def sync_batch(request):
    try: body=json.loads(request.body); items=body.get("items",[])
    except json.JSONDecodeError: return JsonResponse({"error":"invalid_json"},status=400)
    if not isinstance(items,list) or len(items)>20:return JsonResponse({"error":"batch_limit"},status=400)
    accepted=[]
    for item in items:
        item_id=str(item.get("id",""))
        if not item_id: continue
        AuditLog.objects.create(actor_type="sync",actor_id=item_id,action=item.get("action","create"),entity_type=item.get("entity","unknown"),entity_id=item_id)
        accepted.append(item_id)
    return JsonResponse({"accepted":len(accepted),"rejected":len(items)-len(accepted),"idempotency_ids":accepted})

@csrf_exempt
@require_POST
def create_diagnostic_session(request):
    try: body=json.loads(request.body)
    except json.JSONDecodeError:return JsonResponse({"error":"invalid_json"},status=400)
    required=["student_id","started_at","language","device_type"]
    if any(k not in body for k in required):return JsonResponse({"error":"missing_field"},status=400)
    try: student=Student.objects.get(pk=body["student_id"])
    except Student.DoesNotExist:return JsonResponse({"error":"student_not_found"},status=404)
    session=DiagnosticSession.objects.create(student=student,started_at=parse_datetime(body["started_at"]),language=body["language"],device_type=body["device_type"])
    return JsonResponse({"id":session.id,"sync_status":session.sync_status},status=201)