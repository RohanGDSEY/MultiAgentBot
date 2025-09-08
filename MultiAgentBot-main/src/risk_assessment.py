import re
import json
from typing import Dict, List, Any

def calculate_comprehensive_risk_score(person_data: dict, medical_data: dict = None) -> dict:
    """Calculate detailed risk score for underwriting"""
    
    risk_score = 0
    risk_factors = []
    
    # Age-based risk with None handling
    age = person_data.get('age', 35)  # Default to 35 if missing
    if age is None:
        age = 35
        risk_factors.append("Age not specified - using default assessment")
    
    if age > 60:
        risk_score += 4
        risk_factors.append(f"Age {age} - High risk bracket")
    elif age > 50:
        risk_score += 3
        risk_factors.append(f"Age {age} - Medium-high risk")
    elif age > 40:
        risk_score += 2
        risk_factors.append(f"Age {age} - Medium risk")
    elif age > 30:
        risk_score += 1
        risk_factors.append(f"Age {age} - Low-medium risk")
    
    # Medical factors with None handling
    if medical_data:
        bmi = medical_data.get('bmi', None)
        if bmi is not None:
            if bmi > 30:
                risk_score += 3
                risk_factors.append("BMI > 30 (Obese)")
            elif bmi > 25:
                risk_score += 1
                risk_factors.append("BMI > 25 (Overweight)")
        
        # Rest of your medical assessment code...
        medications = medical_data.get('medications', [])
        if any('statin' in str(med).lower() for med in medications):
            risk_score += 2
            risk_factors.append("Cholesterol medication")
        
        family_history = medical_data.get('family_history', [])
        if any('heart' in str(history).lower() for history in family_history):
            risk_score += 2
            risk_factors.append("Family history of cardiac disease")
        
        smoking = medical_data.get('smoking_status', 'unknown')
        if smoking == 'current':
            risk_score += 4
            risk_factors.append("Current smoker")
        elif smoking == 'former':
            risk_score += 1
            risk_factors.append("Former smoker")
        elif smoking == 'never':
            risk_score -= 1
            risk_factors.append("Non-smoker (beneficial)")
    
    # Determine risk level
    if risk_score <= 3:
        risk_level = "Low"
    elif risk_score <= 6:
        risk_level = "Medium" 
    elif risk_score <= 9:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    return {
        "risk_score": max(0, risk_score),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "max_score": 15
    }

def parse_medical_information(content: str) -> dict:
    """Extract medical information from application text"""
    
    medical_info = {
        'conditions': [],
        'medications': [],
        'family_history': [],
        'bmi': None,
        'smoking_status': 'unknown',
        'height': None,
        'weight': None
    }
    
    content_lower = content.lower()
    
    # Extract BMI
    bmi_match = re.search(r'bmi[:\s]*(\d+\.?\d*)', content_lower)
    if bmi_match:
        medical_info['bmi'] = float(bmi_match.group(1))
    
    # Extract height and weight for BMI calculation
    height_match = re.search(r'height[:\s]*(\d+)', content_lower)
    weight_match = re.search(r'weight[:\s]*(\d+)', content_lower)
    
    if height_match and weight_match:
        height_cm = int(height_match.group(1))
        weight_kg = int(weight_match.group(1))
        if not medical_info['bmi']:  # Only calculate if BMI not already present
            medical_info['bmi'] = round(weight_kg / ((height_cm/100) ** 2), 1)
        medical_info['height'] = height_cm
        medical_info['weight'] = weight_kg
    
    # Extract medications
    med_patterns = [
        r'medications?[:\s]*([^\n]+)',
        r'taking[:\s]*([^\n]+)',
        r'prescribed[:\s]*([^\n]+)'
    ]
    
    for pattern in med_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if 'mg' in match.lower() or 'tablet' in match.lower():
                medical_info['medications'].append(match.strip())
    
    # Extract family history
    family_patterns = [
        r'father[:\s]*([^\n]+)',
        r'mother[:\s]*([^\n]+)',
        r'family history[:\s]*([^\n]+)'
    ]
    
    for pattern in family_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            medical_info['family_history'].append(match.strip())
    
    # Extract smoking status
    if 'never smoked' in content_lower:
        medical_info['smoking_status'] = 'never'
    elif 'current smoker' in content_lower or 'smokes' in content_lower:
        medical_info['smoking_status'] = 'current'
    elif 'former smoker' in content_lower or 'quit smoking' in content_lower:
        medical_info['smoking_status'] = 'former'
    
    # Extract conditions
    condition_keywords = ['diabetes', 'hypertension', 'cholesterol', 'heart', 'cancer', 'asthma']
    for keyword in condition_keywords:
        if keyword in content_lower:
            medical_info['conditions'].append(keyword.title())
    
    return medical_info

def make_underwriting_decision(risk_score: int, compliance_status: dict, medical_pending: bool = False) -> dict:
    """Make structured underwriting decision"""
    
    decision = {
        'recommendation': '',
        'conditions': [],
        'premium_adjustment': 0,
        'rationale': [],
        'next_steps': []
    }
    
    # Check for missing critical documents
    missing_docs = compliance_status.get('missing_required', [])
    
    # Decision logic
    if missing_docs:
        decision['recommendation'] = 'POSTPONE'
        decision['conditions'] = [f"Submit missing documents: {', '.join(missing_docs)}"]
        decision['rationale'] = ['Incomplete documentation']
        decision['next_steps'] = ['Collect missing documents', 'Resubmit for review']
    
    elif medical_pending:
        decision['recommendation'] = 'ACCEPT WITH CONDITIONS'
        decision['conditions'] = ['Complete pending medical tests', 'Medical clearance required']
        decision['rationale'] = ['Core documents complete', 'Medical evaluation pending']
        decision['next_steps'] = ['Complete medical tests', 'Submit test results for final approval']
    
    elif risk_score <= 3:
        decision['recommendation'] = 'ACCEPT'
        decision['rationale'] = ['Low risk profile', 'Complete documentation', 'Standard terms applicable']
        decision['next_steps'] = ['Issue policy', 'Send welcome package']
    
    elif risk_score <= 6:
        decision['recommendation'] = 'ACCEPT WITH RATE-UP'
        decision['premium_adjustment'] = 15 + (risk_score - 3) * 5  # 15-25% based on score
        decision['rationale'] = ['Medium risk profile', 'Rate adjustment required for risk mitigation']
        decision['next_steps'] = ['Issue policy with adjusted premium', 'Send rate explanation']
    
    elif risk_score <= 9:
        decision['recommendation'] = 'DECLINE'
        decision['rationale'] = ['High risk profile', 'Exceeds standard acceptance criteria']
        decision['next_steps'] = ['Send decline letter', 'Suggest alternative products if available']
    
    else:
        decision['recommendation'] = 'DECLINE'
        decision['rationale'] = ['Very high risk profile', 'Unacceptable for standard terms']
        decision['next_steps'] = ['Send decline letter', 'No alternative products recommended']
    
    return decision