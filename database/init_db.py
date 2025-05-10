import sqlite3
from pathlib import Path

# 1. Путь к файлу БД
DB_PATH = Path(__file__).parent / "menu.db"

# 2. Схема: таблица items (class_name UNIQUE, price FLOAT)
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL UNIQUE,
    price     REAL NOT NULL
);
"""

DEFAULT_ITEMS = [
    ('apple_pie', 150.00),
    ('baby_back_ribs', 700.00),
    ('baklava', 200.00),
    ('beef_carpaccio', 900.00),
    ('beef_tartare', 800.00),
    ('beet_salad', 250.00),
    ('beetroot_carrot_soup', 200.00),
    ('bibimbap', 450.00),
    ('bread_pudding', 200.00),
    ('breakfast_burrito', 300.00),
    ('bruschetta', 250.00),
    ('caesar_salad', 350.00),
    ('cannoli', 220.00),
    ('caprese_salad', 400.00),
    ('carrot_cake', 300.00),
    ('ceviche', 600.00),
    ('cheese_plate', 500.00),
    ('cheesecake', 300.00),
    ('chicken_curry', 350.00),
    ('chicken_quesadilla', 300.00),
    ('chicken_wings', 350.00),
    ('chocolate_cake', 300.00),
    ('chocolate_mousse', 280.00),
    ('churros', 180.00),
    ('clam_chowder', 400.00),
    ('club_sandwich', 300.00),
    ('crab_cakes', 450.00),
    ('creme_brulee', 320.00),
    ('croque_madame', 350.00),
    ('cup_cakes', 200.00),
    ('deviled_eggs', 200.00),
    ('donuts', 120.00),
    ('dumplings', 250.00),
    ('edamame', 200.00),
    ('eggs_benedict', 400.00),
    ('escargots', 900.00),
    ('falafel', 200.00),
    ('filet_mignon', 1500.00),
    ('fish_and_chips', 400.00),
    ('foie_gras', 1200.00),
    ('french_fries', 150.00),
    ('french_onion_soup', 300.00),
    ('french_toast', 250.00),
    ('fried_calamari', 500.00),
    ('fried_rice', 300.00),
    ('frozen_yogurt', 200.00),
    ('garlic_bread', 150.00),
    ('gnocchi', 350.00),
    ('greek_salad', 300.00),
    ('grilled_cheese_sandwich', 250.00),
    ('grilled_salmon', 800.00),
    ('guacamole', 300.00),
    ('gyoza', 250.00),
    ('hamburger', 350.00),
    ('hot_and_sour_soup', 300.00),
    ('hot_dog', 200.00),
    ('huevos_rancheros', 300.00),
    ('hummus', 200.00),
    ('ice_cream', 150.00),
    ('lasagna', 400.00),
    ('lobster_bisque', 800.00),
    ('lobster_roll_sandwich', 600.00),
    ('macaroni_and_cheese', 300.00),
    ('macarons', 200.00),
    ('miso_soup', 250.00),
    ('mussels', 500.00),
    ('nachos', 350.00),
    ('omelette', 200.00),
    ('onion_rings', 200.00),
    ('oysters', 900.00),
    ('pad_thai', 350.00),
    ('paella', 600.00),
    ('pancakes', 250.00),
    ('panna_cotta', 300.00),
    ('peking_duck', 1200.00),
    ('pho', 350.00),
    ('pizza', 450.00),
    ('pork_chop', 500.00),
    ('poutine', 300.00),
    ('prime_rib', 1200.00),
    ('pulled_pork_sandwich', 400.00),
    ('ramen', 350.00),
    ('ravioli', 400.00),
    ('red_velvet_cake', 300.00),
    ('risotto', 500.00),
    ('samosa', 200.00),
    ('sashimi', 900.00),
    ('scallops', 600.00),
    ('seaweed_salad', 300.00),
    ('shrimp_and_grits', 500.00),
    ('spaghetti_bolognese', 400.00),
    ('spaghetti_carbonara', 400.00),
    ('spring_rolls', 250.00),
    ('steak', 1500.00),
    ('strawberry_shortcake', 300.00),
    ('sushi', 800.00),
    ('tacos', 300.00),
    ('takoyaki', 300.00),
    ('tiramisu', 350.00),
    ('tuna_tartare', 800.00),
    ('waffles', 250.00)
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # создаём таблицу
    c.executescript(SCHEMA)
    # наполняем дефолтными позициями (если ещё нет)
    for name, price in DEFAULT_ITEMS:
        try:
            c.execute("INSERT INTO items (class_name, price) VALUES (?, ?)", (name, price))
        except sqlite3.IntegrityError:
            # уже есть — пропускаем
            pass
    conn.commit()
    conn.close()
    print(f"БД инициализирована: {DB_PATH} (таблица items, {len(DEFAULT_ITEMS)} позиций)")

if __name__ == "__main__":
    init_db()