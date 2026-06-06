"""
Smart Ticket Understanding Engine — Enhanced Synthetic Dataset Generator
Generates ~500 realistic IT service desk tickets with full fields:
  - Category (Incident Type)
  - Priority
  - Department (Route To)
  - Sentiment
  - Root Cause
  - Resolution Steps
  - Ticket Status (Open / Closed / Reopened)
  - Timestamps (created_at, resolved_at)
  - Ticket ID
"""

import csv
import random
import os
from datetime import datetime, timedelta

# ─── TICKET TEMPLATES BY CATEGORY ────────────────────────────────────────────
TEMPLATES = {
    "Network/VPN": {
        "department": "Infra Team",
        "tickets": [
            "Unable to connect to VPN since morning, urgent client call in 20 mins.",
            "VPN keeps disconnecting every 5 minutes, can't attend any meetings.",
            "Wi-Fi is extremely slow on the 3rd floor, affecting the entire team.",
            "Cannot access corporate network from home, VPN shows authentication error.",
            "Network outage in Building B, none of the systems are accessible.",
            "Internet connectivity is down for the whole office since 9 AM.",
            "VPN connection times out after entering credentials.",
            "Firewall is blocking access to our client's portal, need urgent fix.",
            "Unable to connect to the office printer via network.",
            "DNS resolution failing for internal websites.",
            "Getting 'Network unreachable' error when trying to access shared drives.",
            "Multiple users in floor 5 reporting no internet access.",
            "Wi-Fi password changed and nobody was informed, can't connect.",
            "VPN license expired, cannot work remotely anymore.",
            "Latency issues while connecting to US servers through VPN.",
            "Site-to-site VPN tunnel between Mumbai and Pune office is down.",
            "Cannot access internal tools after the network maintenance last night.",
            "Proxy settings seem broken after the last update.",
            "Network switch in server room B2 appears to be malfunctioning.",
            "Bandwidth throttling is affecting video calls with clients.",
            "Need VPN access configured for a new joiner starting tomorrow.",
            "Our team's VLAN seems misconfigured after the weekend changes.",
            "Can someone check the network cables in conference room C?",
            "Need to whitelist a new IP range for our vendor's API.",
            "Port forwarding request for development server on port 8080.",
        ],
        "root_causes": [
            "DNS misconfiguration after recent network update",
            "VPN gateway overloaded due to high concurrent connections",
            "Firewall rule blocking outbound traffic on port 443",
            "DHCP lease exhaustion on the floor's subnet",
            "Network switch firmware bug causing packet drops",
            "ISP upstream link degradation",
            "Expired SSL certificate on VPN concentrator",
            "Incorrect VLAN tagging after maintenance window",
            "Wi-Fi access point hardware failure",
            "Routing table corruption after configuration push",
        ],
        "resolution_steps": [
            "Flushed DNS cache and reconfigured DNS servers to 8.8.8.8 and 8.8.4.4. Verified connectivity restored.",
            "Restarted VPN gateway service and increased connection pool limit from 100 to 250. Monitored for 30 minutes.",
            "Updated firewall ACL rules to allow traffic on ports 443 and 8443. Tested with curl and browser.",
            "Expanded DHCP scope range and reduced lease time from 8 hours to 4 hours. Released stale leases.",
            "Applied firmware patch v2.4.1 to the affected network switch. Rebooted and verified port connectivity.",
            "Contacted ISP support, they rerouted traffic through backup link. Monitored latency for 1 hour.",
            "Renewed SSL certificate on VPN concentrator and restarted the VPN service. Tested remote connectivity.",
            "Corrected VLAN tagging on trunk ports and verified inter-VLAN routing. Tested from affected workstations.",
            "Replaced faulty Wi-Fi access point with spare unit. Configured SSID and verified coverage on floor.",
            "Rolled back routing table to last known good configuration. Verified all routes with traceroute.",
        ],
    },
    "Hardware": {
        "department": "Desktop Support",
        "tickets": [
            "My laptop screen is flickering badly, can't work at all.",
            "Keyboard stopped working suddenly, need a replacement.",
            "Laptop won't turn on even after charging overnight.",
            "Monitor displays blue screen of death every 30 minutes.",
            "Printer on 4th floor is jamming constantly, need repair.",
            "Laptop battery drains in 1 hour, need replacement.",
            "Mouse cursor is lagging and jumping around the screen.",
            "Docking station not detecting external monitors.",
            "Laptop trackpad is unresponsive, using external mouse as workaround.",
            "Headset microphone not being detected in Teams calls.",
            "Desktop fan is making very loud noise, might overheat soon.",
            "USB ports on my laptop are not working anymore.",
            "Need a second monitor for my desk, approved by manager.",
            "Laptop charger is sparking, this is a safety hazard!",
            "Hard drive making clicking noises, worried about data loss.",
            "Webcam quality is very poor, can't be seen in client meetings.",
            "Laptop is overheating and shutting down during video calls.",
            "Need to replace the broken screen on my Dell Latitude.",
            "Keyboard keys are sticking, spilled coffee on it yesterday.",
            "Scanner in HR department is not scanning properly.",
            "My laptop's SSD seems to be failing, very slow boot times.",
            "Request for a new laptop, current one is 5 years old.",
            "Printer toner needs replacement on floor 2 printer.",
            "External hard drive not being recognized by the system.",
            "Need a new power adapter for my ThinkPad, current one broke.",
        ],
        "root_causes": [
            "Faulty display cable connection inside the laptop chassis",
            "Keyboard membrane circuit failure due to liquid damage",
            "Motherboard power regulation IC failure",
            "Corrupted GPU driver causing blue screen crashes",
            "Paper feed roller worn out from extensive use",
            "Battery cell degradation beyond 80% wear level",
            "Wireless mouse receiver USB port malfunction",
            "Docking station firmware incompatible with latest OS update",
            "Trackpad flex cable loose connection",
            "Audio driver conflict with recent Windows update",
        ],
        "resolution_steps": [
            "Opened laptop chassis and reseated display cable. Tested with external monitor to confirm fix. Flickering resolved.",
            "Replaced keyboard with OEM part (P/N: KB-4521). Tested all keys and function keys. Working normally.",
            "Diagnosed motherboard failure. Replaced motherboard under warranty. Restored user data from backup.",
            "Rolled back GPU driver to version 31.0.15.4601. Ran stress test for 2 hours. No BSOD recurrence.",
            "Cleaned paper feed rollers and replaced worn pickup roller assembly. Ran 50-page test print successfully.",
            "Replaced battery with OEM unit (65Wh). Calibrated battery. Now showing 6+ hours estimated life.",
            "Tested USB ports — rear port functional. Replaced wireless receiver. Updated mouse firmware to v2.1.",
            "Updated docking station firmware to v1.4.2. Reinstalled display drivers. Both monitors detected correctly.",
            "Opened laptop and reseated trackpad flex cable. Applied Kapton tape for secure connection. Trackpad responsive.",
            "Uninstalled conflicting Realtek audio driver. Installed Microsoft-signed driver. Microphone detected in Teams.",
        ],
    },
    "Software/Application": {
        "department": "Application Support",
        "tickets": [
            "SAP is crashing every time I try to generate a report.",
            "Need to install Visual Studio Code on my machine.",
            "Excel is freezing when opening large files.",
            "The CRM application is showing error 500 on login.",
            "Adobe Acrobat license has expired, can't edit PDFs.",
            "Windows update broke my Python environment.",
            "Application is throwing 'out of memory' error constantly.",
            "Need Java JDK 17 installed for development work.",
            "Our internal HR portal is loading very slowly today.",
            "Can't install the required software, getting admin permission error.",
            "Zoom keeps crashing during screen sharing.",
            "The inventory management tool is showing incorrect data.",
            "Need to upgrade Microsoft Office to the latest version.",
            "Custom dashboard application is not loading any data since morning.",
            "Browser-based ERP tool is incompatible with Chrome 120.",
            "Salesforce integration is broken after the latest API update.",
            "Jenkins build pipeline has been failing since yesterday.",
            "JIRA is extremely slow, taking 30+ seconds to load each page.",
            "Docker containers keep crashing on the dev server.",
            "Need license renewal for our project management tool.",
            "The automated testing suite is producing false positives.",
            "VS Code extensions not syncing across my devices.",
            "Need to install Postman for API testing work.",
            "Our deployment script is failing with permission errors.",
            "SharePoint site is not syncing documents properly.",
        ],
        "root_causes": [
            "Memory leak in SAP GUI client version 7.70 patch 4",
            "Group policy restricting software installation for standard users",
            "Insufficient RAM allocation for Excel 64-bit with large datasets",
            "CRM backend database connection pool exhausted",
            "License server unreachable due to network segmentation change",
            "Windows update overwrote Python PATH environment variable",
            "Java heap space configured too low in application config",
            "HR portal backend server running at 98% CPU utilization",
            "UAC policy blocking .exe installation from non-approved sources",
            "Zoom GPU acceleration conflicting with integrated graphics driver",
        ],
        "resolution_steps": [
            "Updated SAP GUI to patch 5 which fixes the memory leak. Cleared SAP cache. Tested report generation successfully.",
            "Created software request in ServiceNow. Approved by manager. Deployed VS Code via SCCM. Verified installation.",
            "Increased Excel memory allocation. Enabled 64-bit mode. Tested with 500K row file — opens in 8 seconds.",
            "Restarted CRM application server and increased DB connection pool from 50 to 150. Login working normally.",
            "Updated license server IP in Adobe config. Reactivated license. PDF editing functionality restored.",
            "Restored Python PATH variable. Reinstalled pip packages from requirements.txt. Python environment working.",
            "Increased JVM heap from 512MB to 2GB in app config. Restarted application. No more OOM errors.",
            "Identified runaway background job on HR portal server. Killed process, added resource limits. Load normalized.",
            "Submitted elevated privileges request. Installed software with admin token. Removed temp admin access.",
            "Disabled hardware acceleration in Zoom settings. Updated GPU driver to latest version. Screen sharing stable.",
        ],
    },
    "Account/Access": {
        "department": "IAM Team",
        "tickets": [
            "Locked out of my account after too many wrong password attempts.",
            "Need access to the production database for debugging.",
            "My account got disabled, I am unable to login since morning.",
            "New employee needs Active Directory account and email setup.",
            "MFA is not working, not receiving OTP on my phone.",
            "Need admin access to AWS console for deployment.",
            "Password expired and the reset link is not working.",
            "Cannot access the shared folder, getting permission denied.",
            "Need to revoke access for an employee who left last week.",
            "Single sign-on is not working for Salesforce.",
            "My account is showing as disabled in Active Directory.",
            "Need to set up VPN credentials for a new contractor.",
            "Can't login to the time tracking system with my credentials.",
            "Need read-only access to the finance department's reports.",
            "Guest Wi-Fi access needed for visiting clients next week.",
            "My two-factor authentication app was reset, need to re-enroll.",
            "Need to change my username after a legal name change.",
            "Service account password needs rotation as per policy.",
            "Unable to access the code repository, getting 403 Forbidden.",
            "Need temporary elevated privileges for system maintenance.",
            "Bulk user creation needed for 50 new interns joining Monday.",
            "My account seems compromised, seeing login attempts from unknown IPs.",
            "Need API keys generated for our new microservice.",
            "LDAP authentication failing for our internal application.",
            "Access request for the company's knowledge management portal.",
        ],
        "root_causes": [
            "Account lockout policy triggered after 5 failed login attempts",
            "Access request not processed — missing manager approval in workflow",
            "Account auto-disabled by HR integration due to incorrect termination date",
            "Active Directory replication delay between domain controllers",
            "MFA token seed desynchronized after phone factory reset",
            "IAM role policy missing required permissions for console access",
            "SMTP relay for password reset emails blocked by spam filter",
            "NTFS permission inheritance broken on shared folder",
            "Offboarding workflow incomplete — manual step was missed",
            "SAML assertion attribute mapping mismatch after IdP update",
        ],
        "resolution_steps": [
            "Unlocked account in Active Directory. Reset password. Advised user on password policy (12+ chars, no reuse).",
            "Processed access request with manager's verbal approval. Granted read-only DB access. Logged in audit trail.",
            "Corrected termination date in HR system. Re-enabled AD account. Verified all group memberships restored.",
            "Forced AD replication between DCs using repadmin /syncall. Account visible across all sites within 5 minutes.",
            "Re-enrolled user's MFA device. Generated new QR code. Verified 2FA working with test login.",
            "Updated IAM role to include required EC2 and S3 permissions. User verified console access working.",
            "Whitelisted password reset email domain in spam filter. Resent reset link. User confirmed receipt.",
            "Reset NTFS permissions on shared folder. Re-applied inheritance. User confirmed access to all subfolders.",
            "Completed offboarding checklist. Disabled AD account. Revoked VPN, email, and application access.",
            "Updated SAML attribute mapping in IdP configuration. Tested SSO flow. Salesforce login working.",
        ],
    },
    "Email/Communication": {
        "department": "Collaboration Team",
        "tickets": [
            "Outlook is not syncing emails, last sync was 3 hours ago.",
            "Unable to join Microsoft Teams meeting, getting connection error.",
            "Not receiving emails from external clients since yesterday.",
            "Calendar invites are not showing up in my Outlook.",
            "Teams chat messages are showing as 'sending' but never delivered.",
            "My email mailbox is full, need storage quota increased.",
            "Can't access shared mailbox for our support team.",
            "Teams call quality is terrible, audio keeps cutting out.",
            "Email signature is not appearing when sending from mobile.",
            "Need to set up a distribution list for the new marketing team.",
            "Outlook search is not finding old emails anymore.",
            "Auto-replies are not working for my out-of-office message.",
            "Unable to attach files larger than 10MB in Outlook.",
            "Teams channel notifications are not working on my phone.",
            "Need to recover deleted emails from last week.",
            "Email forwarding rule I set up is not forwarding properly.",
            "Teams video background effects are not available on my device.",
            "Calendar integration between Teams and Outlook is broken.",
            "Need new email alias for the rebranded department name.",
            "Slack integration with our ticketing tool stopped working.",
            "Teams meeting recordings are not saving to SharePoint.",
            "Can someone set up a new Teams channel for Project Athena?",
            "Getting bounce-back errors when emailing our vendor.",
            "Need to migrate old email archives to the new Exchange server.",
            "Outlook rules are duplicating emails into multiple folders.",
        ],
        "root_causes": [
            "Exchange ActiveSync profile corrupted on the device",
            "Teams desktop client cache corrupted after update",
            "External email relay IP blacklisted by recipient's spam filter",
            "Calendar delegate permissions misconfigured",
            "Teams service regional outage affecting message delivery",
            "Mailbox exceeded 50GB quota — archiving policy not applied",
            "Shared mailbox permissions not propagated to user's profile",
            "Network jitter exceeding 150ms causing audio packet loss",
            "Mobile Outlook app using legacy protocol without signature support",
            "Distribution list creation requires Exchange admin privileges",
        ],
        "resolution_steps": [
            "Removed and re-added Exchange account in Outlook. Rebuilt OST file. Email syncing normally now.",
            "Cleared Teams cache from AppData/Microsoft/Teams. Reinstalled Teams. Meeting connection restored.",
            "Submitted delisting request for mail server IP. Configured SPF/DKIM records. External emails flowing.",
            "Reconfigured calendar delegate permissions via Exchange admin center. Calendar invites now visible.",
            "Confirmed Microsoft service incident SI-12345. Workaround: use Teams web. Service restored after 2 hours.",
            "Applied auto-archive policy for emails older than 1 year. Freed 15GB. Mailbox operational.",
            "Added user to shared mailbox via PowerShell Add-MailboxPermission. Restarted Outlook. Access confirmed.",
            "Switched user to wired connection for calls. QoS policy applied for Teams traffic. Audio quality improved.",
            "Updated mobile Outlook app to latest version. Configured signature in app settings. Signature now appears.",
            "Created distribution list via Exchange admin center. Added 25 members. Test email delivered to all.",
        ],
    },
    "Database": {
        "department": "DBA Team",
        "tickets": [
            "Production database is running extremely slow since morning.",
            "Need to restore yesterday's backup for the customer database.",
            "SQL query timeout errors on the reporting dashboard.",
            "Database connection pool is exhausted, app is throwing errors.",
            "Data discrepancy found between production and reporting databases.",
            "Need to create a new database schema for the upcoming project.",
            "MongoDB replica set is out of sync.",
            "Automated database backup failed last night.",
            "Need to optimize slow-running stored procedures.",
            "Database disk usage is at 95%, need urgent cleanup.",
            "Need to set up read replicas for better performance.",
            "Migration script failed midway, database is in inconsistent state.",
            "Need to export data from PostgreSQL to CSV for audit.",
            "Redis cache is not invalidating properly.",
            "Database user permissions need to be audited for compliance.",
            "Table locking issues causing application timeouts.",
            "Need to set up database monitoring and alerts.",
            "Oracle license renewal due next month, need to plan.",
            "Need to implement data masking for PII in test environments.",
            "Deadlock detected in the transaction processing module.",
            "Database replication lag is over 30 seconds.",
            "Need to archive old records older than 3 years.",
            "Connection string update needed after database server migration.",
            "Elasticsearch cluster health is showing yellow status.",
            "Need to set up automated index maintenance schedules.",
        ],
        "root_causes": [
            "Missing index on frequently queried column causing full table scans",
            "Backup job conflicting with nightly ETL process",
            "Unoptimized query with multiple nested subqueries and no limit clause",
            "Application not releasing database connections — connection leak",
            "Replication lag causing stale data on reporting replica",
            "Schema migration tool not available in restricted environment",
            "MongoDB secondary node fell behind due to oplog overflow",
            "Backup storage volume reached capacity — backup job failed silently",
            "Stored procedure using cursor-based iteration instead of set operations",
            "Orphaned temp tables and log files consuming disk space",
        ],
        "resolution_steps": [
            "Added composite index on (customer_id, created_at). Query time reduced from 45s to 0.3s. Monitored for 1 hour.",
            "Restored database from last good backup (23:00 snapshot). Verified data integrity. 0 records lost.",
            "Rewrote query using CTEs and added pagination (LIMIT 1000). Query timeout resolved. Dashboard loading in 2s.",
            "Identified connection leak in application code. Fixed with try-finally block. Pool usage dropped from 100% to 30%.",
            "Restarted replication from snapshot. Verified data consistency with checksum comparison. Lag now under 1 second.",
            "Created schema using DBA admin account. Granted appropriate permissions to development team. Schema ready for use.",
            "Resized oplog to 50GB. Performed initial sync on secondary. Replica set healthy with 0 lag.",
            "Expanded backup storage by 500GB. Cleared old backups beyond retention. Backup job ran successfully.",
            "Rewrote stored procedure using set-based operations. Execution time reduced from 12 minutes to 8 seconds.",
            "Ran cleanup script to remove orphaned temp tables. Purged transaction logs older than 7 days. Freed 200GB.",
        ],
    },
    "Security": {
        "department": "Security Team",
        "tickets": [
            "Received a suspicious phishing email claiming to be from HR.",
            "Antivirus detected malware on my laptop, what should I do?",
            "Unauthorized login attempt detected on my account from Russia.",
            "Need security clearance for a new vendor accessing our systems.",
            "USB drives are not being blocked despite the security policy.",
            "Someone is sending emails from my account without my knowledge.",
            "Found an exposed API endpoint with no authentication.",
            "Our SSL certificate for the customer portal expires tomorrow!",
            "Need vulnerability assessment for our new web application.",
            "Suspicious network traffic detected from a workstation on floor 3.",
            "Data breach suspected — customer data may have been leaked.",
            "Need to implement IP whitelisting for our admin panel.",
            "Ransomware warning message appeared on a screen in accounting.",
            "Security audit findings need to be addressed by end of week.",
            "Need to enable disk encryption on all company laptops.",
            "Penetration testing scheduled for next week, need coordination.",
            "Employee reported clicking on a phishing link, need investigation.",
            "Need to update firewall rules to block known malicious IPs.",
            "Compliance report shows we're not meeting GDPR requirements.",
            "Need to implement DLP policies for email attachments.",
            "Found credentials committed in a public GitHub repository.",
            "Security awareness training completion report needed for audit.",
            "Third-party library vulnerability CVE-2024-XXXX affecting our app.",
            "Need to disable TLS 1.0 and 1.1 on all our web servers.",
            "Incident response plan needs to be updated and tested.",
        ],
        "root_causes": [
            "Phishing email bypassed spam filter due to spoofed domain",
            "User downloaded infected attachment from personal email",
            "Brute force attack from foreign IP — weak password policy",
            "Vendor access request pending security review",
            "USB blocking GPO not applied to the correct OU in AD",
            "Account credentials compromised via credential stuffing attack",
            "Developer deployed API without authentication middleware",
            "SSL certificate auto-renewal job failed due to DNS validation error",
            "New web app not included in regular vulnerability scan scope",
            "Workstation infected with cryptominer via drive-by download",
        ],
        "resolution_steps": [
            "Quarantined phishing email from all mailboxes. Added sender domain to block list. Sent awareness alert to users.",
            "Isolated laptop from network. Ran full malware scan and removed trojan. Re-imaged machine. Restored data from backup.",
            "Blocked IP range in firewall. Force-reset user password. Enabled account lockout policy (5 attempts / 15 min).",
            "Completed vendor security assessment. Granted time-limited VPN access with MFA. Access expires in 30 days.",
            "Corrected GPO scope to include affected OU. Forced gpupdate. Verified USB ports blocked on test workstation.",
            "Reset account password. Revoked all active sessions. Enabled MFA. Reviewed audit logs for unauthorized actions.",
            "Added JWT authentication middleware to API. Deployed fix to production. Verified with Postman — 401 for unauthenticated requests.",
            "Fixed DNS TXT record for domain validation. Renewed SSL cert manually. Configured cert-manager for auto-renewal.",
            "Added web app to vulnerability scanner scope. Ran initial scan. Found 3 medium-risk issues — remediation tickets created.",
            "Isolated workstation. Removed cryptominer process. Blocked C2 server IP in firewall. Patched browser to latest version.",
        ],
    },
    "General IT": {
        "department": "Service Desk",
        "tickets": [
            "How do I set up my new laptop for the first time?",
            "Can someone help me connect to the office projector?",
            "Where can I find the IT policy documents?",
            "Need help setting up my work phone with company email.",
            "Is there a guide for using the new time tracking system?",
            "Request for an IT orientation session for new joiners.",
            "How to access the company intranet from home?",
            "Can I use my personal laptop for office work?",
            "What is the process to request new software?",
            "Need help organizing files on the shared drive.",
            "How do I connect my Bluetooth headset to the laptop?",
            "What are the approved cloud storage solutions?",
            "Can I get a desk phone instead of using my mobile?",
            "How to submit IT expense reimbursement?",
            "Need assistance setting up dual boot on my workstation.",
            "What is the IT equipment return process for leaving employees?",
            "Can you provide details about the company's BYOD policy?",
            "How to configure out-of-office auto-reply in Outlook?",
            "Need a tutorial on using the company's video conferencing tool.",
            "What is the procedure for requesting a new email distribution list?",
            "Need to know the IT asset tagging process.",
            "Where do I find software license information for my tools?",
            "How to map a network drive on my new laptop?",
            "General inquiry about upcoming system maintenance schedule.",
            "Looking for documentation on our internal API standards.",
        ],
        "root_causes": [
            "New employee onboarding documentation not provided during HR induction",
            "Projector HDMI port not compatible with USB-C laptop — adapter needed",
            "IT policy documents moved to new SharePoint site — old link broken",
            "Mobile device management profile not pushed to new phone",
            "Time tracking system user guide not updated after recent upgrade",
            "IT orientation not scheduled by HR for the new batch",
            "VPN required for intranet access from home — user unaware",
            "BYOD policy document not easily discoverable on intranet",
            "Software request form link updated but not communicated",
            "Shared drive folder structure not documented for new team members",
        ],
        "resolution_steps": [
            "Walked user through laptop setup: Wi-Fi, VPN, email, Teams, and printer. Provided setup checklist document.",
            "Provided USB-C to HDMI adapter. Connected to projector. Configured display to extend mode. Working.",
            "Shared updated SharePoint link to IT policy documents. Added shortcut on user's desktop for quick access.",
            "Pushed MDM profile to phone via Intune. Configured company email and Teams. Verified sync working.",
            "Shared updated user guide PDF. Walked user through time entry process. Provided FAQ document.",
            "Scheduled IT orientation session for next Monday. Prepared laptop kits for 10 new joiners.",
            "Configured VPN on user's home laptop. Tested intranet access. Provided VPN usage guide document.",
            "Shared BYOD policy document. Explained personal device registration process via IT portal.",
            "Shared new software request form link. Guided user through the approval workflow steps.",
            "Created folder structure guide for the shared drive. Mapped network drive on user's laptop.",
        ],
    },
}

# ─── PRIORITY INDICATORS ─────────────────────────────────────────────────────
URGENCY_MODIFIERS = {
    "Critical": [
        "URGENT: ", "CRITICAL: ", "EMERGENCY: ", "SYSTEM DOWN: ",
        " — THIS IS BLOCKING ALL WORK!", " We are losing money every minute!",
        " Production is completely down!", " This needs IMMEDIATE attention!",
        " All users are affected, total outage!",
    ],
    "High": [
        "ASAP: ", "High Priority: ", "Important: ",
        " Need this fixed within the hour.", " Can't do any work until this is resolved.",
        " Client demo in 1 hour, please help!", " Deadline is today, really need help.",
        " Boss is escalating this already.",
    ],
    "Medium": [
        " It's causing some inconvenience but I can manage for now.",
        " Would appreciate if this could be fixed today.",
        " Not super urgent but affecting my productivity.",
        " Please look into this when you get a chance.",
        " I have a workaround but it's slowing me down.",
    ],
    "Low": [
        " No rush, whenever you get a chance.",
        " This is just a general question.",
        " Not urgent at all, just planning ahead.",
        " Low priority, can wait until next week.",
        " Just curious about this, no hurry.",
    ],
}

# ─── SENTIMENT PATTERNS ──────────────────────────────────────────────────────
SENTIMENT_MODIFIERS = {
    "Urgent": [
        " Please help ASAP!", " This is extremely urgent!",
        " Can't wait any longer!", " Need immediate assistance!",
        " Every minute counts!", "",
    ],
    "Frustrated": [
        " This has been going on for days and nobody has helped!",
        " I've raised this issue 3 times already, very disappointed.",
        " Really frustrated with the IT support response time.",
        " This is unacceptable, I can't believe this is still broken.",
        " Terrible experience, I expect better from our IT team.",
        "",
    ],
    "Neutral": [
        "", "", "", "", "",
        " Thanks in advance.", " Let me know if you need more info.",
        " Happy to provide more details if needed.",
    ],
    "Positive": [
        " Thanks for always being so helpful!",
        " Appreciate the quick turnaround last time!",
        " Great job on the last fix, hoping for the same this time.",
        " Love how efficient the IT team has been lately!",
        " Just a small request, you guys are doing great work!",
    ],
}


def random_timestamp(base_date, max_days_back=90):
    """Generate a random timestamp within max_days_back days of base_date."""
    delta = timedelta(
        days=random.randint(0, max_days_back),
        hours=random.randint(8, 18),  # business hours
        minutes=random.randint(0, 59),
    )
    return base_date - delta


def generate_dataset(num_tickets=5000, output_path="data/tickets.csv"):
    """Generate a balanced synthetic dataset of IT support tickets."""
    categories = list(TEMPLATES.keys())
    priorities = ["Critical", "High", "Medium", "Low"]
    sentiments = ["Urgent", "Frustrated", "Neutral", "Positive"]

    # Weight distributions for realistic data
    priority_weights = [0.10, 0.25, 0.40, 0.25]
    sentiment_weights = [0.15, 0.20, 0.50, 0.15]

    # Status weights: 70% Closed, 15% Open, 15% Reopened
    statuses = ["Closed", "Open", "Reopened"]
    status_weights = [0.70, 0.15, 0.15]

    tickets = []
    per_category = num_tickets // len(categories)
    base_date = datetime(2026, 6, 5, 17, 0, 0)  # reference date
    ticket_counter = 0

    for category in categories:
        dept = TEMPLATES[category]["department"]
        base_tickets = TEMPLATES[category]["tickets"]
        root_causes = TEMPLATES[category]["root_causes"]
        resolutions = TEMPLATES[category]["resolution_steps"]

        for i in range(per_category):
            ticket_counter += 1
            ticket_id = f"TKT-{ticket_counter:04d}"

            # Pick base ticket (cycle through templates)
            base = base_tickets[i % len(base_tickets)]

            # Assign priority & sentiment with weighted randomness
            priority = random.choices(priorities, weights=priority_weights, k=1)[0]
            sentiment = random.choices(sentiments, weights=sentiment_weights, k=1)[0]

            # Adjust: Critical priority often comes with Urgent sentiment
            if priority == "Critical" and random.random() > 0.3:
                sentiment = "Urgent"
            elif priority == "Low" and random.random() > 0.4:
                sentiment = "Neutral"

            # Build the full ticket text
            prefix = ""
            suffix = ""
            mod = random.choice(URGENCY_MODIFIERS[priority])
            if mod.startswith(" "):
                suffix += mod
            else:
                prefix = mod

            sent_mod = random.choice(SENTIMENT_MODIFIERS[sentiment])
            suffix += sent_mod

            ticket_text = prefix + base + suffix

            # Generate timestamps
            created_at = random_timestamp(base_date)

            # Assign status
            status = random.choices(statuses, weights=status_weights, k=1)[0]

            # Resolution details
            root_cause = random.choice(root_causes)
            resolution = random.choice(resolutions)

            # Generate resolved_at based on status and priority
            resolved_at = None
            if status in ("Closed", "Reopened"):
                # Resolution time varies by priority
                resolution_hours = {
                    "Critical": random.uniform(0.25, 2.0),
                    "High": random.uniform(1.0, 4.0),
                    "Medium": random.uniform(2.0, 12.0),
                    "Low": random.uniform(4.0, 48.0),
                }
                hours = resolution_hours[priority]
                resolved_at = created_at + timedelta(hours=hours)
            else:
                # Open tickets have no resolution yet
                root_cause = "Investigation in progress"
                resolution = "Pending — ticket assigned to engineer"

            # Determine recommended action
            action = get_recommended_action(category, priority, sentiment)

            tickets.append({
                "ticket_id": ticket_id,
                "ticket_text": ticket_text.strip(),
                "category": category,
                "priority": priority,
                "department": dept,
                "sentiment": sentiment,
                "root_cause": root_cause,
                "resolution_steps": resolution,
                "ticket_status": status,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "resolved_at": resolved_at.strftime("%Y-%m-%d %H:%M:%S") if resolved_at else "",
                "recommended_action": action,
            })

    # Shuffle for good measure
    random.shuffle(tickets)

    # Write CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fieldnames = [
        "ticket_id", "ticket_text", "category", "priority",
        "department", "sentiment", "root_cause", "resolution_steps",
        "ticket_status", "created_at", "resolved_at", "recommended_action",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tickets)

    print(f"✅ Generated {len(tickets)} tickets → {output_path}")
    return tickets


def get_recommended_action(category, priority, sentiment):
    """Rule-based action recommendation based on category, priority, and sentiment."""
    dept = TEMPLATES[category]["department"]

    if priority == "Critical":
        return f"IMMEDIATE ESCALATION to {dept} — Page on-call engineer"
    if priority == "High":
        if sentiment in ("Urgent", "Frustrated"):
            return f"Priority escalation to {dept} — Assign senior engineer"
        return f"Assign to {dept} — Target 2-hour resolution"
    if priority == "Medium":
        if sentiment == "Frustrated":
            return f"Assign to {dept} — Follow up with user, target 4-hour resolution"
        return f"Queue for {dept} — Standard SLA (8 hours)"
    # Low priority
    if sentiment == "Positive":
        return f"Queue for {dept} — Respond within 24 hours with appreciation"
    return f"Queue for {dept} — Respond within 24 hours"


if __name__ == "__main__":
    random.seed(42)
    generate_dataset(5000, os.path.join(os.path.dirname(__file__), "tickets.csv"))
