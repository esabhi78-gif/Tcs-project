"""
Action Recommendation Engine for the Smart Ticket Understanding Engine.
Maps (category, priority, sentiment) → recommended next action.
"""

# ─── DEPARTMENT MAPPING ──────────────────────────────────────────────────────
CATEGORY_TO_DEPARTMENT = {
    "Network/VPN": "Infra Team",
    "Hardware": "Desktop Support",
    "Software/Application": "Application Support",
    "Account/Access": "IAM Team",
    "Email/Communication": "Collaboration Team",
    "Database": "DBA Team",
    "Security": "Security Team",
    "General IT": "Service Desk",
}

# ─── ACTION TEMPLATES ────────────────────────────────────────────────────────
ACTION_TEMPLATES = {
    "Critical": {
        "Urgent":     "🚨 IMMEDIATE ESCALATION — Page on-call {dept} engineer. Notify incident commander. Target: 15 min response.",
        "Frustrated": "🚨 IMMEDIATE ESCALATION — Page on-call {dept} engineer. Assign customer liaison for communication.",
        "Neutral":    "🚨 IMMEDIATE ESCALATION — Page on-call {dept} engineer. Target: 30 min response.",
        "Positive":   "🚨 IMMEDIATE ESCALATION — Page on-call {dept} engineer. Target: 30 min response.",
    },
    "High": {
        "Urgent":     "⚡ Priority assignment to {dept} — Assign senior engineer. Target: 1-hour resolution. Send acknowledgment immediately.",
        "Frustrated": "⚡ Priority assignment to {dept} — Assign senior engineer. Call user to acknowledge & provide ETA.",
        "Neutral":    "⚡ Assign to {dept} — Target 2-hour resolution. Send automated acknowledgment.",
        "Positive":   "⚡ Assign to {dept} — Target 2-hour resolution. Thank user for patience.",
    },
    "Medium": {
        "Urgent":     "📋 Queue for {dept} — Prioritize in next sprint. Target: 4-hour resolution. Acknowledge urgency.",
        "Frustrated": "📋 Queue for {dept} — Assign & follow up with user within 1 hour. Target: 4-hour resolution.",
        "Neutral":    "📋 Queue for {dept} — Standard SLA (8 hours). Send automated acknowledgment.",
        "Positive":   "📋 Queue for {dept} — Standard SLA (8 hours). Positive interaction noted.",
    },
    "Low": {
        "Urgent":     "📝 Queue for {dept} — Respond within 8 hours. Clarify urgency with user.",
        "Frustrated": "📝 Queue for {dept} — Respond within 8 hours. Reach out to user proactively.",
        "Neutral":    "📝 Queue for {dept} — Respond within 24 hours. Standard process.",
        "Positive":   "📝 Queue for {dept} — Respond within 24 hours. Acknowledge positive feedback.",
    },
}


def get_recommended_action(category, priority, sentiment):
    """
    Generate a recommended action based on ticket classification results.
    
    Args:
        category: Incident type (e.g., 'Network/VPN', 'Hardware')
        priority: Priority level ('Critical', 'High', 'Medium', 'Low')
        sentiment: Detected sentiment ('Urgent', 'Frustrated', 'Neutral', 'Positive')
    
    Returns:
        Recommended action string with department-specific routing.
    """
    dept = CATEGORY_TO_DEPARTMENT.get(category, "Service Desk")

    # Get action template
    priority_actions = ACTION_TEMPLATES.get(priority, ACTION_TEMPLATES["Medium"])
    template = priority_actions.get(sentiment, priority_actions["Neutral"])

    return template.format(dept=dept)


def get_department(category):
    """Get the routing department for a given incident category."""
    return CATEGORY_TO_DEPARTMENT.get(category, "Service Desk")


def get_sla_hours(priority):
    """Get the SLA target hours for a given priority level."""
    sla_map = {
        "Critical": 0.5,   # 30 minutes
        "High": 2.0,       # 2 hours
        "Medium": 8.0,     # 8 hours (1 business day)
        "Low": 24.0,       # 24 hours
    }
    return sla_map.get(priority, 8.0)
