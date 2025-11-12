import requests

def get_furniture_suggestions(style="Modern", room="Living Room"):
    """
    Mock furniture suggestions.
    Replace this function with API calls to IKEA / Wayfair.
    """
    sample_items = [
        {
            "name": "Minimalist Sofa",
            "price": "$499",
            "url": "https://www.ikea.com/us/en/p/klippan-sofa-00278801/",
            "img": "https://www.ikea.com/ext/ikea/img/klippan.jpg"
        },
        {
            "name": "Wooden Coffee Table",
            "price": "$199",
            "url": "https://www.wayfair.com/furniture/pdp/coffee-table-12345.html",
            "img": "https://example.com/coffee_table.jpg"
        },
        {
            "name": "Floor Lamp",
            "price": "$129",
            "url": "https://www.ikea.com/us/en/p/hektar-floor-lamp-20390394/",
            "img": "https://www.ikea.com/ext/ikea/img/lamp.jpg"
        }
    ]
    return sample_items
