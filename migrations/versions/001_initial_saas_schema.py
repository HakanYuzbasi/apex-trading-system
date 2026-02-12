"""Initial SaaS schema - users, subscriptions, roles, api_keys, feature_access, service_jobs

Revision ID: 001
Revises: None
Create Date: 2026-02-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("mfa_enabled", sa.Boolean(), default=False, nullable=False),
        sa.Column("mfa_secret", sa.String(64), nullable=True),
        sa.Column("oauth_provider", sa.String(50), nullable=True),
        sa.Column("oauth_id", sa.String(255), nullable=True),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_oauth", "users", ["oauth_provider", "oauth_id"], unique=True)

    # --- subscriptions ---
    op.create_table(
        "subscriptions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("tier", sa.String(20), default="free", nullable=False),
        sa.Column("status", sa.String(50), default="active", nullable=False),
        sa.Column("stripe_subscription_id", sa.String(255), nullable=True, unique=True),
        sa.Column("stripe_price_id", sa.String(255), nullable=True),
        sa.Column("current_period_start", sa.DateTime(), nullable=True),
        sa.Column("current_period_end", sa.DateTime(), nullable=True),
        sa.Column("cancel_at_period_end", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # --- user_roles ---
    op.create_table(
        "user_roles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(50), nullable=False),
    )
    op.create_index("ix_user_roles_unique", "user_roles", ["user_id", "role"], unique=True)

    # --- api_keys ---
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key_hash", sa.String(255), unique=True, nullable=False),
        sa.Column("key_prefix", sa.String(20), nullable=False),
        sa.Column("label", sa.String(100), nullable=True),
        sa.Column("permissions", sa.JSON(), default=list),
        sa.Column("rate_limit_rpm", sa.Integer(), default=60),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])

    # --- feature_access ---
    op.create_table(
        "feature_access",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tier", sa.String(20), nullable=False),
        sa.Column("feature_key", sa.String(100), nullable=False),
        sa.Column("enabled", sa.Boolean(), default=True, nullable=False),
        sa.Column("rate_limit_daily", sa.Integer(), default=-1),
    )
    op.create_index("ix_feature_access_tier_key", "feature_access", ["tier", "feature_key"], unique=True)

    # Seed feature access matrix
    op.execute("""
        INSERT INTO feature_access (id, tier, feature_key, enabled, rate_limit_daily) VALUES
        -- Backtest Validator
        ('fa-bv-basic', 'basic', 'backtest_validator', true, 10),
        ('fa-bv-pro', 'pro', 'backtest_validator', true, 100),
        ('fa-bv-ent', 'enterprise', 'backtest_validator', true, -1),
        -- Execution Simulator
        ('fa-es-basic', 'basic', 'execution_simulator', true, 10),
        ('fa-es-pro', 'pro', 'execution_simulator', true, 100),
        ('fa-es-ent', 'enterprise', 'execution_simulator', true, -1),
        -- Drift Monitor
        ('fa-dm-pro', 'pro', 'drift_monitor', true, -1),
        ('fa-dm-ent', 'enterprise', 'drift_monitor', true, -1),
        -- Portfolio Allocator
        ('fa-pa-pro', 'pro', 'portfolio_allocator', true, 50),
        ('fa-pa-ent', 'enterprise', 'portfolio_allocator', true, -1),
        -- Compliance Copilot
        ('fa-cc-ent', 'enterprise', 'compliance_copilot', true, 50)
    """)

    # --- service_jobs ---
    op.create_table(
        "service_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("feature_key", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), default="pending", nullable=False),
        sa.Column("input_params", sa.JSON(), default=dict),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("result_file_path", sa.String(500), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_service_jobs_user_feature", "service_jobs", ["user_id", "feature_key"])


def downgrade() -> None:
    op.drop_table("service_jobs")
    op.drop_table("feature_access")
    op.drop_table("api_keys")
    op.drop_table("user_roles")
    op.drop_table("subscriptions")
    op.drop_table("users")
