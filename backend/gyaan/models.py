from django.db import models
class School(models.Model):
 name=models.CharField(max_length=200);district=models.CharField(max_length=100);block=models.CharField(max_length=100);created_at=models.DateTimeField(auto_now_add=True)
class Student(models.Model):
 school=models.ForeignKey(School,on_delete=models.PROTECT);external_id=models.CharField(max_length=64);class_no=models.PositiveSmallIntegerField();gender=models.CharField(max_length=20,blank=True);medium=models.CharField(max_length=20);parent_phone_hash=models.CharField(max_length=128,blank=True);created_at=models.DateTimeField(auto_now_add=True)
 class Meta:constraints=[models.UniqueConstraint(fields=["school","external_id"],name="uniq_student_school_external")]
class DiagnosticSession(models.Model):
 student=models.ForeignKey(Student,on_delete=models.CASCADE);started_at=models.DateTimeField();completed_at=models.DateTimeField(null=True,blank=True);language=models.CharField(max_length=5);device_type=models.CharField(max_length=32);sync_status=models.CharField(max_length=16,default="pending")
class DiagnosticResponse(models.Model):
 session=models.ForeignKey(DiagnosticSession,on_delete=models.CASCADE);question_id=models.CharField(max_length=64);selected_option=models.CharField(max_length=64);response_time_ms=models.PositiveIntegerField(null=True);skipped=models.BooleanField(default=False);synced_at=models.DateTimeField(auto_now_add=True)
 class Meta:constraints=[models.UniqueConstraint(fields=["session","question_id"],name="uniq_response")]
class AuditLog(models.Model):
 actor_type=models.CharField(max_length=20);actor_id=models.CharField(max_length=64,blank=True);action=models.CharField(max_length=100);entity_type=models.CharField(max_length=100);entity_id=models.CharField(max_length=64);timestamp=models.DateTimeField(auto_now_add=True)