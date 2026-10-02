from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [("gyaan", "0002_learning_domain")]

    operations = [
        migrations.AlterField(
            model_name="diagnosticresponse",
            name="response_time_ms",
            field=models.PositiveIntegerField(null=True),
        ),
    ]
