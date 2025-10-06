import numpy as np
from typing import Dict, Tuple, List


# EASI component categorization
EASI_CATEGORIES = {
    'erythema': {
        'name': 'Erythema (Redness)',
        'conditions': [
            'Post-Inflammatory hyperpigmentation',
            'Erythema ab igne', 'Erythema annulare centrifugum',
            'Erythema elevatum diutinum', 'Erythema gyratum repens',
            'Erythema multiforme', 'Erythema nodosum',
            'Flagellate erythema', 'Annular erythema',
            'Drug Rash', 'Allergic Contact Dermatitis',
            'Irritant Contact Dermatitis', 'Contact dermatitis',
            'Acute dermatitis', 'Chronic dermatitis',
            'Acute and chronic dermatitis',
            'Sunburn', 'Photodermatitis', 'Phytophotodermatitis',
            'Rosacea', 'Seborrheic Dermatitis', 'Stasis Dermatitis',
            'Perioral Dermatitis',
            'Burn erythema of abdominal wall',
            'Burn erythema of back of hand',
            'Burn erythema of lower leg',
            'Cellulitis', 'Infection of skin', 'Viral Exanthem',
            'Infected eczema', 'Crusted eczematous dermatitis',
            'Inflammatory dermatosis',
            'Vasculitis of the skin', 'Leukocytoclastic Vasculitis',
            'Cutaneous lupus',
            'CD - Contact dermatitis',
            'Acute dermatitis, NOS',
            'Herpes Simplex',
            'Hypersensitivity',
            'Impetigo',
            'Pigmented purpuric eruption',
            'Pityriasis rosea',
            'Tinea',
            'Tinea Versicolor'
        ]
    },
    'induration': {
        'name': 'Induration/Papulation (Swelling/Bumps)',
        'conditions': [
            'Prurigo nodularis', 'Urticaria', 'Granuloma annulare', 'Morphea',
            'Scleroderma', 'Lichen Simplex Chronicus',
            'Lichen planus', 'lichenoid eruption',
            'Lichen nitidus', 'Lichen spinulosus', 'Lichen striatus',
            'Keratosis pilaris', 'Molluscum Contagiosum',
            'Verruca vulgaris', 'Folliculitis', 'Acne',
            'Hidradenitis', 'Nodular vasculitis', 'Sweet syndrome',
            'Necrobiosis lipoidica', 'Basal Cell Carcinoma',
            'SCC', 'SCCIS', 'SK', 'ISK',
            'Cutaneous T Cell Lymphoma', 'Skin cancer',
            'Adnexal neoplasm', 'Insect Bite', 'Milia',
            'Miliaria', 'Xanthoma', 'Psoriasis',
            'Lichen planus/lichenoid eruption'
        ]
    },
    'excoriation': {
        'name': 'Excoriation (Scratching Damage)',
        'conditions': [
            'Inflicted skin lesions',
            'Scabies', 'Abrasion', 'Abrasion of wrist',
            'Superficial wound of body region', 'Scrape',
            'Animal bite - wound', 'Pruritic dermatitis',
            'Prurigo', 'Atopic dermatitis',
            'Scab'
        ]
    },
    'lichenification': {
        'name': 'Lichenification (Skin Thickening)',
        'conditions': [
            'Lichenified eczematous dermatitis',
            'Acanthosis nigricans', 'Hyperkeratosis of skin',
            'HK - Hyperkeratosis', 'Keratoderma',
            'Ichthyosis', 'Ichthyosiform dermatosis',
            'Chronic eczema', 'Psoriasis',
            'Xerosis'
        ]
    }
}


def probability_to_score(prob: float) -> int:
    """Convert probability to EASI score (0-3)"""
    if prob < 0.171:
        return 0
    elif prob < 0.238:
        return 1
    elif prob < 0.421:
        return 2
    elif prob < 0.614:
        return 3
    else:
        return 3


def calculate_easi_scores(predictions: Dict) -> Tuple[Dict, int]:
    """
    Calculate EASI component scores based on condition probabilities
    
    Args:
        predictions: Dictionary containing prediction results
    
    Returns:
        Tuple of (easi_results dict, total_easi_score int)
    """
    easi_results = {}
    all_condition_probs = predictions['all_condition_probabilities']
    
    for component, category_info in EASI_CATEGORIES.items():
        # Find all conditions in this category
        category_conditions = []
        
        for condition_name, probability in all_condition_probs.items():
            # Skip "Eczema" as it should not be included
            if condition_name.lower() == 'eczema':
                continue
            
            # Check if condition is in category
            if condition_name in category_info['conditions']:
                individual_score = probability_to_score(probability)
                if individual_score > 0:
                    category_conditions.append({
                        'condition': condition_name,
                        'probability': probability,
                        'individual_score': individual_score
                    })
        
        # Sort by probability
        category_conditions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Calculate component score (sum, capped at 3)
        component_score = sum(c['individual_score'] for c in category_conditions)
        component_score = min(component_score, 3)
        
        easi_results[component] = {
            'name': category_info['name'],
            'score': component_score,
            'contributing_conditions': category_conditions
        }
    
    # Calculate total EASI score
    total_easi = sum(result['score'] for result in easi_results.values())
    
    return easi_results, total_easi


def format_easi_response(easi_results: Dict, total_easi: int) -> Dict:
    """
    Format EASI results for API response
    
    Args:
        easi_results: EASI calculation results
        total_easi: Total EASI score
    
    Returns:
        Formatted dictionary for JSON response
    """
    return {
        'total_score': total_easi,
        'components': {
            'erythema': easi_results['erythema']['score'],
            'induration': easi_results['induration']['score'],
            'excoriation': easi_results['excoriation']['score'],
            'lichenification': easi_results['lichenification']['score']
        },
        'severity': get_severity_level(total_easi),
        'component_details': {
            component: {
                'name': data['name'],
                'score': data['score'],
                'contributing_conditions': [
                    {
                        'condition': c['condition'],
                        'probability': round(c['probability'], 4),
                        'contribution': c['individual_score']
                    }
                    for c in data['contributing_conditions']
                ]
            }
            for component, data in easi_results.items()
        }
    }


def get_severity_level(total_easi: int) -> str:
    """Get severity level description from EASI score"""
    if total_easi == 0:
        return "No significant EASI features detected"
    elif total_easi <= 3:
        return "Mild EASI severity"
    elif total_easi <= 6:
        return "Moderate EASI severity"
    elif total_easi <= 9:
        return "Severe EASI severity"
    else:
        return "Very Severe EASI severity"