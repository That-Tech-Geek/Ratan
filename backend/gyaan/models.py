from django.db import models

class School(models.Model):
    name=models.CharField(max_length=200);district=models.CharField(max_length=100);block=models.CharField(max_length=100);created_at=models.DateTimeField(auto_now_add=True)

class Student(models.Model):
    school=models.ForeignKey(School,on_delete=models.PROTECT);external_id=models.CharField(max_length=64);class_no=models.PositiveSmallIntegerField();gender=models.CharField(max_length=20,blank=True);medium=models.CharField(max_length=20);parent_phone_hash=models.CharField(max_length=128,blank=True);created_at=models.DateTimeField(auto_now_add=True)
    class Meta: constraints=[models.UniqueConstraint(fields=["school","external_id"],name="uniq_student_school_external")]

class ConsentRecord(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);consent_type=models.CharField(max_length=40);consent_text_version=models.CharField(max_length=40);consented_at=models.DateTimeField();verification_method=models.CharField(max_length=40);withdrawn_at=models.DateTimeField(null=True,blank=True)

class DiagnosticSession(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);started_at=models.DateTimeField();completed_at=models.DateTimeField(null=True,blank=True);language=models.CharField(max_length=5);device_type=models.CharField(max_length=32);sync_status=models.CharField(max_length=16,default="pending")

class DiagnosticResponse(models.Model):
    session=models.ForeignKey(DiagnosticSession,on_delete=models.CASCADE);question_id=models.CharField(max_length=64);selected_option=models.CharField(max_length=64);response_time_ms=models.PositiveIntegerField(null=True);skipped=models.BooleanField(default=False);synced_at=models.DateTimeField(auto_now_add=True)
    class Meta: constraints=[models.UniqueConstraint(fields=["session","question_id"],name="uniq_response")]

class LikertSession(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);started_at=models.DateTimeField();completed_at=models.DateTimeField(null=True,blank=True);language=models.CharField(max_length=5);sync_status=models.CharField(max_length=16,default="pending")

class LikertResponse(models.Model):
    session=models.ForeignKey(LikertSession,on_delete=models.CASCADE);item_id=models.CharField(max_length=64);score=models.PositiveSmallIntegerField();skipped=models.BooleanField(default=False);synced_at=models.DateTimeField(auto_now_add=True)
    class Meta: constraints=[models.UniqueConstraint(fields=["session","item_id"],name="uniq_likert_response")]

class LearningPreference(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);visual_score=models.FloatField();auditory_score=models.FloatField();reading_writing_score=models.FloatField();kinesthetic_score=models.FloatField();top_preference=models.CharField(max_length=30);second_preference=models.CharField(max_length=30);is_mixed=models.BooleanField(default=False);computed_at=models.DateTimeField(auto_now_add=True)

class GapReport(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);session=models.ForeignKey(DiagnosticSession,on_delete=models.CASCADE);weak_topics=models.JSONField(default=list);strong_topics=models.JSONField(default=list);suggested_next_step=models.TextField();generated_at=models.DateTimeField(auto_now_add=True);pdf_url=models.URLField(blank=True)

class RecheckSession(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);original_session=models.ForeignKey(DiagnosticSession,on_delete=models.PROTECT);started_at=models.DateTimeField();completed_at=models.DateTimeField(null=True,blank=True);improvement_json=models.JSONField(default=dict);sync_status=models.CharField(max_length=16,default="pending")

class GuidanceCard(models.Model):
    student=models.ForeignKey(Student,on_delete=models.CASCADE);aspiration=models.CharField(max_length=300,blank=True);obstacle=models.CharField(max_length=500,blank=True);next_step_options=models.JSONField(default=list);reviewed_by_teacher=models.BooleanField(default=False);generated_at=models.DateTimeField(auto_now_add=True)

class AuditLog(models.Model):
    actor_type=models.CharField(max_length=20);actor_id=models.CharField(max_length=64,blank=True);action=models.CharField(max_length=100);entity_type=models.CharField(max_length=100);entity_id=models.CharField(max_length=64);timestamp=models.DateTimeField(auto_now_add=True)
