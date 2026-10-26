"""
Dataset Generation and Seeding Script for E-Commerce Platform.
Creates products, users, categories, interactions, and ratings.
Exports raw CSVs to data/raw/ and populates SQLite/MySQL database.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Ensure root directory is on PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import init_db, SessionLocal, Product, User, Category, Interaction, Rating, Wishlist

# Deterministic seed for reproducible evaluation and tests
random.seed(42)
np.random.seed(42)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

CATEGORIES_DATA = [
    {"category_id": 1, "category_name": "Electronics", "slug": "electronics", "icon": "laptop"},
    {"category_id": 2, "category_name": "Fashion", "slug": "fashion", "icon": "shirt"},
    {"category_id": 3, "category_name": "Home & Kitchen", "slug": "home-kitchen", "icon": "home"},
    {"category_id": 4, "category_name": "Sports & Fitness", "slug": "sports-fitness", "icon": "dumbbell"},
    {"category_id": 5, "category_name": "Beauty & Personal Care", "slug": "beauty", "icon": "sparkles"},
    {"category_id": 6, "category_name": "Books & Stationery", "slug": "books", "icon": "book-open"},
]

# Curated catalog of products with realistic attributes, features, and price points
PRODUCTS_CATALOG = [
    # Electronics - Smartphones
    {
        "product_id": "PROD_ELEC_001",
        "product_name": "Apple iPhone 15 (128 GB) - Black",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "Apple",
        "description": "Dynamic Island 48MP main camera with 2x Telephoto, USB-C connectivity, durable color-infused glass and aluminum design, A16 Bionic chip with 5-core GPU.",
        "price": 69999.0,
        "rating": 4.6,
        "review_count": 3420,
        "discount_percent": 12,
        "popularity_score": 95.5,
        "image_url": "https://images.unsplash.com/photo-1592750475338-74b7b21085ab?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_002",
        "product_name": "Apple iPhone 15 Plus (128 GB) - Blue",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "Apple",
        "description": "Large 6.7-inch Super Retina XDR display, Dynamic Island, all-day battery life up to 26 hours video playback, 48MP camera, A16 Bionic chip.",
        "price": 79999.0,
        "rating": 4.7,
        "review_count": 1820,
        "discount_percent": 10,
        "popularity_score": 88.0,
        "image_url": "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_003",
        "product_name": "Apple iPhone 14 (128 GB) - Midnight",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "Apple",
        "description": "6.1-inch Super Retina XDR display, advanced dual-camera system with Photonic Engine, Action mode, Crash Detection, A15 Bionic chip with 5-core GPU.",
        "price": 57999.0,
        "rating": 4.5,
        "review_count": 5210,
        "discount_percent": 17,
        "popularity_score": 91.2,
        "image_url": "https://images.unsplash.com/photo-1591337676887-a217a6970a8a?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_004",
        "product_name": "Samsung Galaxy S24 5G (Onyx Black, 256GB)",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "Samsung",
        "description": "Galaxy AI powered smartphone, Circle to Search, Live Translate, Note Assist, 50MP triple camera system, Snapdragon 8 Gen 3, 120Hz Dynamic AMOLED 2X.",
        "price": 74999.0,
        "rating": 4.6,
        "review_count": 2890,
        "discount_percent": 14,
        "popularity_score": 89.4,
        "image_url": "https://images.unsplash.com/photo-1610945415295-d9bbf067e59c?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_005",
        "product_name": "Google Pixel 8 (Hazel, 128GB)",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "Google",
        "description": "Google Tensor G3 chip, advanced AI computational photography with Best Take and Audio Magic Eraser, 6.2-inch Actua display 120Hz, 7 years OS updates.",
        "price": 63999.0,
        "rating": 4.4,
        "review_count": 1430,
        "discount_percent": 15,
        "popularity_score": 82.5,
        "image_url": "https://images.unsplash.com/photo-1598327105666-5b89351aff97?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_006",
        "product_name": "OnePlus 12 (Silky Black, 256GB)",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "OnePlus",
        "description": "Snapdragon 8 Gen 3, 4th Gen Hasselblad Camera system with 64MP 3X periscope telephoto, 5400mAh battery with 100W SUPERVOOC charging, 2K 120Hz ProXDR display.",
        "price": 64999.0,
        "rating": 4.5,
        "review_count": 2100,
        "discount_percent": 8,
        "popularity_score": 84.1,
        "image_url": "https://images.unsplash.com/photo-1580910051074-3eb694886505?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_007",
        "product_name": "Xiaomi Redmi Note 13 Pro+ 5G (Fusion Purple, 256GB)",
        "category": "Electronics",
        "subcategory": "Smartphones",
        "brand": "Xiaomi",
        "description": "Budget flagship killer: 200MP OIS camera, 3D curved 1.5K 120Hz AMOLED display, IP68 water resistance, 120W HyperCharge, Dimensity 7200-Ultra.",
        "price": 31999.0,
        "rating": 4.3,
        "review_count": 4120,
        "discount_percent": 18,
        "popularity_score": 79.0,
        "image_url": "https://images.unsplash.com/photo-1567581935884-3349723552ca?w=600&auto=format&fit=crop&q=80"
    },

    # Electronics - Laptops & Computing
    {
        "product_id": "PROD_ELEC_008",
        "product_name": "Apple MacBook Air 15-inch M3 (16GB RAM, 512GB SSD)",
        "category": "Electronics",
        "subcategory": "Laptops",
        "brand": "Apple",
        "description": "Blazing-fast M3 chip with 8-core CPU and 10-core GPU, Liquid Retina display, 18 hours battery life, 1080p FaceTime HD camera, MagSafe 3 charging, midnight finish.",
        "price": 134900.0,
        "rating": 4.8,
        "review_count": 1450,
        "discount_percent": 7,
        "popularity_score": 93.8,
        "image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_009",
        "product_name": "Dell XPS 13 OLED (Intel Core Ultra 7, 16GB, 1TB SSD)",
        "category": "Electronics",
        "subcategory": "Laptops",
        "brand": "Dell",
        "description": "Ultra-thin CNC aluminum laptop with 3K OLED InfinityEdge touch display, Intel AI Boost NPU, seamless glass touchpad, backlit capacitive touch function row.",
        "price": 142990.0,
        "rating": 4.5,
        "review_count": 890,
        "discount_percent": 10,
        "popularity_score": 81.3,
        "image_url": "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_010",
        "product_name": "HP Pavilion 15 (AMD Ryzen 7, 16GB RAM, 512GB SSD)",
        "category": "Electronics",
        "subcategory": "Laptops",
        "brand": "HP",
        "description": "All-round productivity laptop with 15.6-inch FHD micro-edge display, Audio by B&O, fast charging, Wi-Fi 6, Windows 11 Home with MS Office.",
        "price": 62990.0,
        "rating": 4.3,
        "review_count": 2780,
        "discount_percent": 16,
        "popularity_score": 77.4,
        "image_url": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_011",
        "product_name": "Lenovo IdeaPad Slim 3 (Intel Core i5, 16GB, 512GB)",
        "category": "Electronics",
        "subcategory": "Laptops",
        "brand": "Lenovo",
        "description": "Smart everyday lightweight laptop, military-grade durability MIL-STD-810H, rapid charge technology, privacy webcam shutter, Dolby Audio speakers.",
        "price": 49990.0,
        "rating": 4.2,
        "review_count": 3120,
        "discount_percent": 22,
        "popularity_score": 75.0,
        "image_url": "https://images.unsplash.com/photo-1541807084-5c52b6b3adef?w=600&auto=format&fit=crop&q=80"
    },

    # Electronics - Audio & Accessories
    {
        "product_id": "PROD_ELEC_012",
        "product_name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "category": "Electronics",
        "subcategory": "Audio",
        "brand": "Sony",
        "description": "Industry leading noise cancellation with two processors and 8 microphones, Auto NC Optimizer, Hi-Res Audio wireless, 30-hour battery life with quick charging.",
        "price": 29990.0,
        "rating": 4.7,
        "review_count": 4820,
        "discount_percent": 15,
        "popularity_score": 92.5,
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_013",
        "product_name": "Apple AirPods Pro (2nd Generation) with USB-C MagSafe",
        "category": "Electronics",
        "subcategory": "Audio",
        "brand": "Apple",
        "description": "H2 chip delivering up to 2x more Active Noise Cancellation, Adaptive Audio, Personalized Spatial Audio with dynamic head tracking, dust/sweat/water resistance.",
        "price": 24900.0,
        "rating": 4.8,
        "review_count": 6120,
        "discount_percent": 5,
        "popularity_score": 94.0,
        "image_url": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_014",
        "product_name": "boAt Rockerz 450 Bluetooth Wireless On-Ear Headphones",
        "category": "Electronics",
        "subcategory": "Audio",
        "brand": "boAt",
        "description": "Budget bass headphones: Up to 15 hours playtime, 40mm dynamic drivers, padded ear cushions, dual modes bluetooth & aux, integrated controls.",
        "price": 1499.0,
        "rating": 4.1,
        "review_count": 18200,
        "discount_percent": 62,
        "popularity_score": 85.0,
        "image_url": "https://images.unsplash.com/photo-1484704849700-f032a568e944?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_015",
        "product_name": "Logitech MX Master 3S Wireless Performance Mouse",
        "category": "Electronics",
        "subcategory": "Peripherals",
        "brand": "Logitech",
        "description": "Ergonomic quiet clicks mouse, 8K DPI any-surface tracking, MagSpeed electromagnetic scrolling, USB-C rechargeable, multi-device flow control.",
        "price": 8995.0,
        "rating": 4.7,
        "review_count": 3940,
        "discount_percent": 18,
        "popularity_score": 87.2,
        "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_ELEC_016",
        "product_name": "Keychron K2 Wireless Mechanical Keyboard (RGB Backlit)",
        "category": "Electronics",
        "subcategory": "Peripherals",
        "brand": "Keychron",
        "description": "75% layout compact Bluetooth/Wired mechanical keyboard for Mac and Windows, hot-swappable Gateron G Pro Brown switches, 4000mAh battery.",
        "price": 7999.0,
        "rating": 4.6,
        "review_count": 1640,
        "discount_percent": 12,
        "popularity_score": 80.5,
        "image_url": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=600&auto=format&fit=crop&q=80"
    },

    # Fashion - Men & Women Apparel
    {
        "product_id": "PROD_FASH_001",
        "product_name": "Nike Dri-FIT Men's Running T-Shirt",
        "category": "Fashion",
        "subcategory": "Activewear",
        "brand": "Nike",
        "description": "Breathable moisture-wicking fabric engineered with Dri-FIT technology. Reflective elements for low-light visibility, standard fit for relaxed athletic comfort.",
        "price": 1895.0,
        "rating": 4.5,
        "review_count": 2850,
        "discount_percent": 20,
        "popularity_score": 86.4,
        "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FASH_002",
        "product_name": "Adidas Originals Men's Trefoil Hoodie",
        "category": "Fashion",
        "subcategory": "Casual Wear",
        "brand": "Adidas",
        "description": "Classic pullover heavyweight French terry cotton hoodie with embroidered Trefoil chest logo, kangaroo pouch pocket, ribbed cuffs and hem.",
        "price": 4599.0,
        "rating": 4.6,
        "review_count": 1980,
        "discount_percent": 30,
        "popularity_score": 88.0,
        "image_url": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FASH_003",
        "product_name": "Levi's Men's 511 Slim Fit Jeans",
        "category": "Fashion",
        "subcategory": "Denim",
        "brand": "Levi's",
        "description": "Iconic slim fit cut with room to move. Premium stretch denim, zip fly with button closure, classic 5-pocket styling in dark indigo wash.",
        "price": 3299.0,
        "rating": 4.4,
        "review_count": 4120,
        "discount_percent": 25,
        "popularity_score": 83.5,
        "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FASH_004",
        "product_name": "Zara Oversized Textured Knit Sweater",
        "category": "Fashion",
        "subcategory": "Knitwear",
        "brand": "Zara",
        "description": "Contemporary round neckline sweater with long sleeves and ribbed trim. Soft blend yarn with waffle knit texture, relaxed drop-shoulder silhouette.",
        "price": 3990.0,
        "rating": 4.3,
        "review_count": 920,
        "discount_percent": 15,
        "popularity_score": 76.8,
        "image_url": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FASH_005",
        "product_name": "Puma Men's Training Track Pants",
        "category": "Fashion",
        "subcategory": "Activewear",
        "brand": "Puma",
        "description": "Performance track pants with dryCELL moisture control fabric, zippered side pockets, tapered athletic cut, elasticated waistband with drawstring.",
        "price": 2299.0,
        "rating": 4.2,
        "review_count": 1650,
        "discount_percent": 35,
        "popularity_score": 79.2,
        "image_url": "https://images.unsplash.com/photo-1552902865-b72c031ac5ea?w=600&auto=format&fit=crop&q=80"
    },

    # Fashion - Footwear & Sneakers
    {
        "product_id": "PROD_FASH_006",
        "product_name": "Nike Air Jordan 1 Low Retro (White/University Blue)",
        "category": "Fashion",
        "subcategory": "Footwear",
        "brand": "Nike",
        "description": "Timeless basketball streetwear heritage with premium leather upper, encapsulated Air-Sole heel unit, rubber cupsole with durable traction.",
        "price": 8995.0,
        "rating": 4.8,
        "review_count": 3560,
        "discount_percent": 5,
        "popularity_score": 94.5,
        "image_url": "https://images.unsplash.com/photo-1552346154-21d32810aba3?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FASH_007",
        "product_name": "Adidas Ultraboost Light Running Shoes",
        "category": "Fashion",
        "subcategory": "Footwear",
        "brand": "Adidas",
        "description": "30% lighter BOOST midsole material delivering epic energy return, Continental rubber outsole grip, Primeknit+ foot-hugging upper.",
        "price": 14999.0,
        "rating": 4.7,
        "review_count": 2180,
        "discount_percent": 25,
        "popularity_score": 90.1,
        "image_url": "https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FASH_008",
        "product_name": "Puma Smash V2 Leather Casual Sneakers",
        "category": "Fashion",
        "subcategory": "Footwear",
        "brand": "Puma",
        "description": "Clean tennis-inspired silhouette with soft leather upper, SoftFoam+ comfort sockliner, durable vulcanized rubber outsole for all-day wear.",
        "price": 2799.0,
        "rating": 4.3,
        "review_count": 5890,
        "discount_percent": 45,
        "popularity_score": 84.8,
        "image_url": "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=600&auto=format&fit=crop&q=80"
    },

    # Home & Kitchen
    {
        "product_id": "PROD_HOME_001",
        "product_name": "Instant Pot Duo 7-in-1 Electric Pressure Cooker (6 Quart)",
        "category": "Home & Kitchen",
        "subcategory": "Kitchen Appliances",
        "brand": "Instant Pot",
        "description": "Multi-functional cooker: Pressure cooker, slow cooker, rice cooker, steamer, sauté pan, yogurt maker and food warmer with 13 one-touch Smart Programs.",
        "price": 9999.0,
        "rating": 4.7,
        "review_count": 6400,
        "discount_percent": 20,
        "popularity_score": 91.0,
        "image_url": "https://images.unsplash.com/photo-1544233726-9f1d2b27be8b?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_HOME_002",
        "product_name": "Philips Digital Air Fryer HD9252/90 (4.1 Litre)",
        "category": "Home & Kitchen",
        "subcategory": "Kitchen Appliances",
        "brand": "Philips",
        "description": "Rapid Air Technology frying with up to 90% less fat. Touch screen with 7 presets, keep warm function, dishwasher safe removable parts.",
        "price": 7999.0,
        "rating": 4.6,
        "review_count": 5210,
        "discount_percent": 30,
        "popularity_score": 89.3,
        "image_url": "https://images.unsplash.com/photo-1585515320310-259814833e62?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_HOME_003",
        "product_name": "Dyson V12 Detect Slim Cordless Vacuum Cleaner",
        "category": "Home & Kitchen",
        "subcategory": "Cleaning Appliances",
        "brand": "Dyson",
        "description": "Laser reveals microscopic dust on hard floors, piezo sensor counts dust particles, Hyperdymium motor spins up to 125,000rpm, up to 60 min runtime.",
        "price": 54900.0,
        "rating": 4.7,
        "review_count": 1420,
        "discount_percent": 12,
        "popularity_score": 86.2,
        "image_url": "https://images.unsplash.com/photo-1558317374-067fb5f30001?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_HOME_004",
        "product_name": "Prestige Iris Plus 750W Mixer Grinder with 4 Jars",
        "category": "Home & Kitchen",
        "subcategory": "Kitchen Appliances",
        "brand": "Prestige",
        "description": "Heavy-duty 750W motor with 3 stainless steel multipurpose jars and 1 transparent juicer jar with blade. Ergonomic handles and overload protection.",
        "price": 3199.0,
        "rating": 4.2,
        "review_count": 9800,
        "discount_percent": 48,
        "popularity_score": 83.1,
        "image_url": "https://images.unsplash.com/photo-1574269909862-7e1d70bb8078?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_HOME_005",
        "product_name": "Nespresso Vertuo Pop Coffee Machine",
        "category": "Home & Kitchen",
        "subcategory": "Coffee & Tea",
        "brand": "Nespresso",
        "description": "Centrifusion extraction technology reads barcode on capsule to tailor brewing. 4 cup sizes from Espresso to Mug, ultra-compact design, fast 30s heat-up.",
        "price": 14999.0,
        "rating": 4.5,
        "review_count": 1890,
        "discount_percent": 15,
        "popularity_score": 80.4,
        "image_url": "https://images.unsplash.com/photo-1517668808822-9ebb02f2a0e6?w=600&auto=format&fit=crop&q=80"
    },

    # Sports & Fitness
    {
        "product_id": "PROD_FIT_001",
        "product_name": "Apple Watch Series 9 GPS 45mm (Midnight Aluminum)",
        "category": "Sports & Fitness",
        "subcategory": "Fitness Trackers",
        "brand": "Apple",
        "description": "S9 SiP chip with Double Tap gesture, brighter 2000 nits display, ECG app, blood oxygen tracking, cycle tracking, crash detection, workout metrics.",
        "price": 44900.0,
        "rating": 4.8,
        "review_count": 3210,
        "discount_percent": 8,
        "popularity_score": 95.0,
        "image_url": "https://images.unsplash.com/photo-1508685096489-7aacd43bd3b1?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FIT_002",
        "product_name": "Garmin Forerunner 265 GPS Running Smartwatch",
        "category": "Sports & Fitness",
        "subcategory": "Fitness Trackers",
        "brand": "Garmin",
        "description": "Vibrant AMOLED touchscreen display, training readiness score, morning report, wrist-based running power and dynamics, up to 13 days battery life.",
        "price": 46990.0,
        "rating": 4.7,
        "review_count": 940,
        "discount_percent": 10,
        "popularity_score": 84.6,
        "image_url": "https://images.unsplash.com/photo-1579586337278-3befd40fd17a?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FIT_003",
        "product_name": "Fitbit Charge 6 Fitness & Health Tracker",
        "category": "Sports & Fitness",
        "subcategory": "Fitness Trackers",
        "brand": "Fitbit",
        "description": "Built-in GPS, EDA stress scan, ECG heart monitor, 40+ workout modes, YouTube Music and Google Maps controls, 7 days battery life.",
        "price": 14999.0,
        "rating": 4.3,
        "review_count": 2150,
        "discount_percent": 18,
        "popularity_score": 81.2,
        "image_url": "https://images.unsplash.com/photo-1575311373937-040b8e1fd5b6?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FIT_004",
        "product_name": "Decathlon Domyos 20kg Adjustable Cast Iron Dumbbell Set",
        "category": "Sports & Fitness",
        "subcategory": "Strength Training",
        "brand": "Decathlon",
        "description": "Versatile home gym cast iron weights set with threaded chrome bars, spinlock collars, sturdy compact carry case for biceps, triceps, chest, and shoulder workouts.",
        "price": 4999.0,
        "rating": 4.6,
        "review_count": 3890,
        "discount_percent": 15,
        "popularity_score": 82.0,
        "image_url": "https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_FIT_005",
        "product_name": "Boldfit Yoga Mat with Alignment Lines (6mm Non-Slip)",
        "category": "Sports & Fitness",
        "subcategory": "Yoga & Pilates",
        "brand": "Boldfit",
        "description": "Eco-friendly high-density TPE cushioning mat with laser-engraved alignment guide marks, waterproof sweat-resistant grip, carrying strap included.",
        "price": 1299.0,
        "rating": 4.4,
        "review_count": 4870,
        "discount_percent": 40,
        "popularity_score": 78.5,
        "image_url": "https://images.unsplash.com/photo-1601925260368-ae2f83cf8b7f?w=600&auto=format&fit=crop&q=80"
    },

    # Beauty & Personal Care
    {
        "product_id": "PROD_BEAUTY_001",
        "product_name": "Cetaphil Gentle Skin Cleanser (500ml)",
        "category": "Beauty & Personal Care",
        "subcategory": "Skincare",
        "brand": "Cetaphil",
        "description": "Hydrating dermatologist-recommended face wash for sensitive and dry skin, formulated with Niacinamide, Vitamin B5, and hydrating glycerin, soap-free fragrance-free.",
        "price": 1150.0,
        "rating": 4.7,
        "review_count": 12400,
        "discount_percent": 15,
        "popularity_score": 92.0,
        "image_url": "https://images.unsplash.com/photo-1556228720-195a672e8a03?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_BEAUTY_002",
        "product_name": "The Ordinary Niacinamide 10% + Zinc 1% Serum (30ml)",
        "category": "Beauty & Personal Care",
        "subcategory": "Skincare",
        "brand": "The Ordinary",
        "description": "High-strength vitamin and mineral blemish formula that reduces the appearance of skin blemishes, congestion, and balances visible sebum activity.",
        "price": 600.0,
        "rating": 4.6,
        "review_count": 15800,
        "discount_percent": 0,
        "popularity_score": 93.4,
        "image_url": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_BEAUTY_003",
        "product_name": "L'Oreal Paris Revitalift 1.5% Hyaluronic Acid Serum",
        "category": "Beauty & Personal Care",
        "subcategory": "Skincare",
        "brand": "L'Oreal",
        "description": "Intensive plumping face serum with micro hyaluronic acid molecules for radiant youthful glowing skin. Lightweight non-greasy fast absorbing.",
        "price": 999.0,
        "rating": 4.4,
        "review_count": 6700,
        "discount_percent": 25,
        "popularity_score": 85.0,
        "image_url": "https://images.unsplash.com/photo-1608248597359-561307b22596?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_BEAUTY_004",
        "product_name": "Philips AquaTouch S1323/45 Wet & Dry Cordless Shaver",
        "category": "Beauty & Personal Care",
        "subcategory": "Grooming",
        "brand": "Philips",
        "description": "ComfortCut self-sharpening blades with 3-directional flex heads, pop-up trimmer for mustache and sideburns, 45 minutes cordless shaving per 1 hr charge.",
        "price": 2495.0,
        "rating": 4.3,
        "review_count": 4150,
        "discount_percent": 30,
        "popularity_score": 81.0,
        "image_url": "https://images.unsplash.com/photo-1621607512214-68297480165e?w=600&auto=format&fit=crop&q=80"
    },

    # Books & Stationery
    {
        "product_id": "PROD_BOOK_001",
        "product_name": "Atomic Habits by James Clear (Hardcover)",
        "category": "Books & Stationery",
        "subcategory": "Self-Help",
        "brand": "Penguin",
        "description": "An easy and proven way to build good habits and break bad ones. International bestseller on behavioral science, systems thinking, and personal mastery.",
        "price": 699.0,
        "rating": 4.9,
        "review_count": 28400,
        "discount_percent": 25,
        "popularity_score": 97.2,
        "image_url": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_BOOK_002",
        "product_name": "Designing Data-Intensive Applications by Martin Kleppmann",
        "category": "Books & Stationery",
        "subcategory": "Computer Science",
        "brand": "O'Reilly",
        "description": "The big ideas behind reliable, scalable, and maintainable systems. Essential reading for software engineers, backend developers, and data architects.",
        "price": 1850.0,
        "rating": 4.9,
        "review_count": 5600,
        "discount_percent": 15,
        "popularity_score": 92.1,
        "image_url": "https://images.unsplash.com/photo-1532012164546-f432f2e3edd4?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_BOOK_003",
        "product_name": "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow",
        "category": "Books & Stationery",
        "subcategory": "Computer Science",
        "brand": "O'Reilly",
        "description": "Aurélien Géron guides you through neural networks, deep learning architectures, end-to-end ML pipelines, transformers, and practical industry workflows.",
        "price": 2250.0,
        "rating": 4.8,
        "review_count": 3400,
        "discount_percent": 18,
        "popularity_score": 90.0,
        "image_url": "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=600&auto=format&fit=crop&q=80"
    },
    {
        "product_id": "PROD_BOOK_004",
        "product_name": "Moleskine Classic Notebook (Hard Cover, Ruled/Lined, Black)",
        "category": "Books & Stationery",
        "subcategory": "Stationery",
        "brand": "Moleskine",
        "description": "Premium acid-free ivory pages, rounded corners, elastic closure, bookmark ribbon, expandable inner back pocket. Perfect for journaling and sketching.",
        "price": 1795.0,
        "rating": 4.6,
        "review_count": 3100,
        "discount_percent": 10,
        "popularity_score": 83.0,
        "image_url": "https://images.unsplash.com/photo-1589829085413-56de8ae18c73?w=600&auto=format&fit=crop&q=80"
    }
]

# User Personas for rich scenario testing and live demo
SEED_USERS = [
    {
        "user_id": "USER_TECH_001",
        "username": "Arjun Sharma",
        "email": "arjun.tech@example.com",
        "preferred_category": "Electronics"
    },
    {
        "user_id": "USER_FASH_002",
        "username": "Priya Kapoor",
        "email": "priya.fashion@example.com",
        "preferred_category": "Fashion"
    },
    {
        "user_id": "USER_HOME_003",
        "username": "Sneha Roy",
        "email": "sneha.home@example.com",
        "preferred_category": "Home & Kitchen"
    },
    {
        "user_id": "USER_FIT_004",
        "username": "Vikram Malhotra",
        "email": "vikram.fitness@example.com",
        "preferred_category": "Sports & Fitness"
    },
    {
        "user_id": "USER_DEV_005",
        "username": "Aditya Singh",
        "email": "aditya.dev@example.com",
        "preferred_category": "Books & Stationery"
    },
    {
        "user_id": "USER_COLD_006",
        "username": "Rohan Verma (New User)",
        "email": "rohan.new@example.com",
        "preferred_category": None
    }
]


def generate_synthetic_interactions(num_additional_users: int = 40):
    """Generates cohesive interaction trails with realistic user behavior clusters."""
    users = list(SEED_USERS)
    for i in range(7, num_additional_users + 7):
        cat = random.choice([c["category_name"] for c in CATEGORIES_DATA])
        users.append({
            "user_id": f"USER_SYNTH_{i:03d}",
            "username": f"User {i}",
            "email": f"user{i}@example.com",
            "preferred_category": cat
        })

    interactions = []
    ratings = []
    wishlists = []

    # Map products by category
    prod_by_cat = {}
    for p in PRODUCTS_CATALOG:
        prod_by_cat.setdefault(p["category"], []).append(p)

    all_prod_ids = [p["product_id"] for p in PRODUCTS_CATALOG]

    # Generate interactions for users (except cold-start user USER_COLD_006)
    base_time = datetime.utcnow() - timedelta(days=30)

    for user in users:
        u_id = user["user_id"]
        pref_cat = user["preferred_category"]

        if u_id == "USER_COLD_006" or pref_cat is None:
            # Keep empty for clean cold start demo
            continue

        # Target products in their preference category (70% probability) vs random (30%)
        num_sessions = random.randint(5, 18)
        interacted_in_session = set()

        for s in range(num_sessions):
            if random.random() < 0.75 and pref_cat in prod_by_cat and prod_by_cat[pref_cat]:
                target_prod = random.choice(prod_by_cat[pref_cat])
            else:
                target_prod = random.choice(PRODUCTS_CATALOG)

            p_id = target_prod["product_id"]
            session_time = base_time + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # View is always recorded
            interactions.append({
                "user_id": u_id,
                "product_id": p_id,
                "interaction_type": "view",
                "timestamp": session_time
            })

            # Click
            if random.random() < 0.65:
                interactions.append({
                    "user_id": u_id,
                    "product_id": p_id,
                    "interaction_type": "click",
                    "timestamp": session_time + timedelta(seconds=random.randint(5, 45))
                })

            # Wishlist / Like
            if random.random() < 0.35:
                interactions.append({
                    "user_id": u_id,
                    "product_id": p_id,
                    "interaction_type": "wishlist",
                    "timestamp": session_time + timedelta(seconds=random.randint(50, 90))
                })
                if p_id not in [w["product_id"] for w in wishlists if w["user_id"] == u_id]:
                    wishlists.append({
                        "user_id": u_id,
                        "product_id": p_id,
                        "added_at": session_time
                    })

            # Cart
            if random.random() < 0.25:
                interactions.append({
                    "user_id": u_id,
                    "product_id": p_id,
                    "interaction_type": "cart",
                    "timestamp": session_time + timedelta(seconds=random.randint(100, 180))
                })

                # Purchase (conversion)
                if random.random() < 0.60:
                    interactions.append({
                        "user_id": u_id,
                        "product_id": p_id,
                        "interaction_type": "purchase",
                        "timestamp": session_time + timedelta(seconds=random.randint(190, 300))
                    })
                    # Rating after purchase
                    if random.random() < 0.70:
                        user_rating = round(random.choice([4.0, 4.5, 5.0, 3.5, 4.0, 5.0]), 1)
                        ratings.append({
                            "user_id": u_id,
                            "product_id": p_id,
                            "rating": user_rating,
                            "review_text": f"Great product! Matches description well. Loved the quality.",
                            "timestamp": session_time + timedelta(days=random.randint(1, 3))
                        })

    return users, interactions, ratings, wishlists


def seed_database_and_export_csv():
    """Initializes tables, seeds DB, and exports raw CSV files."""
    print("🚀 Initializing database schema...")
    init_db()

    users_list, interactions_list, ratings_list, wishlists_list = generate_synthetic_interactions()

    # 1. Export Raw CSVs for ML Pipeline & Notebooks
    df_products = pd.DataFrame(PRODUCTS_CATALOG)
    df_products["created_at"] = datetime.utcnow()
    df_products.to_csv(os.path.join(RAW_DATA_DIR, "products.csv"), index=False)

    df_interactions = pd.DataFrame(interactions_list)
    df_interactions.to_csv(os.path.join(RAW_DATA_DIR, "interactions.csv"), index=False)

    df_users = pd.DataFrame(users_list)
    df_users.to_csv(os.path.join(RAW_DATA_DIR, "users.csv"), index=False)

    print(f"✅ Exported CSV files:")
    print(f"   - {len(df_products)} products -> data/raw/products.csv")
    print(f"   - {len(df_interactions)} interactions -> data/raw/interactions.csv")
    print(f"   - {len(df_users)} users -> data/raw/users.csv")

    # 2. Populate Database via SessionLocal
    db = SessionLocal()
    try:
        # Categories
        for cat_data in CATEGORIES_DATA:
            if not db.query(Category).filter(Category.category_name == cat_data["category_name"]).first():
                db.add(Category(**cat_data))

        # Products
        for prod_data in PRODUCTS_CATALOG:
            existing = db.query(Product).filter(Product.product_id == prod_data["product_id"]).first()
            if not existing:
                db.add(Product(**prod_data))

        # Users
        for user_data in users_list:
            existing = db.query(User).filter(User.user_id == user_data["user_id"]).first()
            if not existing:
                db.add(User(**user_data))

        db.commit()

        # Wishlists
        for w in wishlists_list:
            existing = db.query(Wishlist).filter(
                Wishlist.user_id == w["user_id"],
                Wishlist.product_id == w["product_id"]
            ).first()
            if not existing:
                db.add(Wishlist(**w))

        # Interactions
        if db.query(Interaction).count() == 0:
            for item in interactions_list:
                db.add(Interaction(**item))

        # Ratings
        for r in ratings_list:
            existing = db.query(Rating).filter(
                Rating.user_id == r["user_id"],
                Rating.product_id == r["product_id"]
            ).first()
            if not existing:
                db.add(Rating(**r))

        db.commit()
        print("✅ Database successfully populated with initial data and interaction history!")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database_and_export_csv()
