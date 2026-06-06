"""
Smart Ticket Understanding Engine — Metrics Calculator
Calculates MTTR, MTTM, and other SLA reports from dataset.
"""

import pandas as pd
import numpy as np

def calculate_metrics(csv_path):
    """
    Load dataset and calculate MTTR, MTTM, and resolution rates.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
        
    # Ensure datetime parsing
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    if 'resolved_at' in df.columns:
        # replace empty strings with NaT
        df['resolved_at'] = pd.to_datetime(df['resolved_at'].replace('', np.nan))
        
    metrics = {
        "total_tickets": len(df),
        "status_distribution": df['ticket_status'].value_counts().to_dict() if 'ticket_status' in df.columns else {},
        "category_distribution": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
        "priority_distribution": df['priority'].value_counts().to_dict() if 'priority' in df.columns else {}
    }
    
    # Calculate MTTR (Mean Time To Resolve)
    if 'created_at' in df.columns and 'resolved_at' in df.columns:
        resolved_tickets = df[df['resolved_at'].notna()].copy()
        if not resolved_tickets.empty:
            resolved_tickets['resolution_time_hours'] = (resolved_tickets['resolved_at'] - resolved_tickets['created_at']).dt.total_seconds() / 3600
            
            # Overall MTTR
            metrics['overall_mttr_hours'] = round(resolved_tickets['resolution_time_hours'].mean(), 2)
            
            # MTTR by Category
            mttr_by_cat = resolved_tickets.groupby('category')['resolution_time_hours'].mean().round(2).to_dict()
            metrics['mttr_by_category'] = mttr_by_cat
            
            # MTTR by Priority
            mttr_by_pri = resolved_tickets.groupby('priority')['resolution_time_hours'].mean().round(2).to_dict()
            metrics['mttr_by_priority'] = mttr_by_pri
        else:
            metrics['overall_mttr_hours'] = 0
            metrics['mttr_by_category'] = {}
            metrics['mttr_by_priority'] = {}
            
    # Calculate MTTM (Mean Time To Mitigate - simplified as 30% of MTTR for synthetic data)
    if 'overall_mttr_hours' in metrics:
        metrics['overall_mttm_hours'] = round(metrics['overall_mttr_hours'] * 0.3, 2)
        
    # Resolution Success Rate (Tickets Closed without Reopening)
    if 'ticket_status' in df.columns:
        total_closed = len(df[df['ticket_status'] == 'Closed'])
        total_reopened = len(df[df['ticket_status'] == 'Reopened'])
        total_resolved_attempts = total_closed + total_reopened
        
        if total_resolved_attempts > 0:
            success_rate = (total_closed / total_resolved_attempts) * 100
            metrics['resolution_success_rate'] = round(success_rate, 2)
        else:
            metrics['resolution_success_rate'] = 0
            
    return metrics
    
def get_dataframe(csv_path):
    """Return raw dataframe for custom plotting in Streamlit."""
    try:
        df = pd.read_csv(csv_path)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        if 'resolved_at' in df.columns:
            df['resolved_at'] = pd.to_datetime(df['resolved_at'].replace('', np.nan))
        
        if 'created_at' in df.columns and 'resolved_at' in df.columns:
             df['resolution_time_hours'] = (df['resolved_at'] - df['created_at']).dt.total_seconds() / 3600
             
        return df
    except:
        return None
