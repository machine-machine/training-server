"""Initial schema — 4 tables for training server.

Revision ID: 001
Create Date: 2026-02-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # training_datasets (created first — referenced by training_jobs)
    op.create_table(
        "training_datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("file_format", sa.String(20), nullable=False),
        sa.Column("row_count", sa.BigInteger),
        sa.Column("feature_count", sa.Integer),
        sa.Column("label_column", sa.String(100)),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("metadata", JSONB),
    )
    op.create_index("idx_datasets_name", "training_datasets", ["name"])
    op.create_index("idx_datasets_uploaded_at", "training_datasets", ["uploaded_at"])

    # training_jobs
    op.create_table(
        "training_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("hyperparams", JSONB),
        sa.Column("gpu_id", sa.Integer),
        sa.Column("worker_id", sa.String(100)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("error_message", sa.Text),
        sa.Column("metrics", JSONB),
        sa.Column("dataset_id", UUID(as_uuid=True), sa.ForeignKey("training_datasets.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_jobs_status", "training_jobs", ["status"])
    op.create_index("idx_jobs_type_status", "training_jobs", ["job_type", "status"])
    op.create_index("idx_jobs_created_at", "training_jobs", ["created_at"])

    # model_artifacts
    op.create_table(
        "model_artifacts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("training_jobs.id")),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("sha256_checksum", sa.String(64), nullable=False),
        sa.Column("metrics", JSONB),
        sa.Column("feature_signature", JSONB),
        sa.Column("promoted_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_artifacts_model_type_version", "model_artifacts", ["model_type", "version"], unique=True)

    # training_logs
    op.create_table(
        "training_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("training_jobs.id"), nullable=False),
        sa.Column("level", sa.String(10), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("metadata", JSONB),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_logs_job_id", "training_logs", ["job_id"])
    op.create_index("idx_logs_timestamp", "training_logs", ["timestamp"])


def downgrade() -> None:
    op.drop_table("training_logs")
    op.drop_table("model_artifacts")
    op.drop_table("training_jobs")
    op.drop_table("training_datasets")
