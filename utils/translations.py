"""
Translation utilities for the application.
Provides dictionaries for translating between French and English date elements.
"""

# Dictionary for weekday translations (English -> French)
JOURS_SEMAINE = {
    "Monday": "Lundi",
    "Tuesday": "Mardi",
    "Wednesday": "Mercredi",
    "Thursday": "Jeudi",
    "Friday": "Vendredi",
    "Saturday": "Samedi",
    "Sunday": "Dimanche"
}

# Dictionary for month translations (English -> French)
MOIS_ANNEE = {
    "January": "Janvier",
    "February": "Février",
    "March": "Mars",
    "April": "Avril",
    "May": "Mai",
    "June": "Juin",
    "July": "Juillet",
    "August": "Août",
    "September": "Septembre",
    "October": "Octobre",
    "November": "Novembre",
    "December": "Décembre"
}

# Inverse dictionaries for reverse translations (French -> English)
JOURS_SEMAINE_INVERSE = {v: k for k, v in JOURS_SEMAINE.items()}
MOIS_ANNEE_INVERSE = {v: k for k, v in MOIS_ANNEE.items()}

# Order of days for sorting
JOURS_ORDRE = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def get_french_weekday(english_day):
    """
    Convert English weekday name to French.
    
    Args:
        english_day (str): English weekday name
        
    Returns:
        str: French weekday name or original if not found
    """
    return JOURS_SEMAINE.get(english_day, english_day)


def get_english_weekday(french_day):
    """
    Convert French weekday name to English.
    
    Args:
        french_day (str): French weekday name
        
    Returns:
        str: English weekday name or original if not found
    """
    return JOURS_SEMAINE_INVERSE.get(french_day, french_day)


def get_french_month(english_month):
    """
    Convert English month name to French.
    
    Args:
        english_month (str): English month name
        
    Returns:
        str: French month name or original if not found
    """
    return MOIS_ANNEE.get(english_month, english_month)


def get_english_month(french_month):
    """
    Convert French month name to English.
    
    Args:
        french_month (str): French month name
        
    Returns:
        str: English month name or original if not found
    """
    return MOIS_ANNEE_INVERSE.get(french_month, french_month)
