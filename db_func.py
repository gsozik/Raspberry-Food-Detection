import sqlite3


def get_price(class_name: str, DB_PATH):
    """
    Возвращает цену блюда по его class_name из таблицы items.
    Если блюдо не найдено — возвращает None.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT price FROM items WHERE class_name = ?",
            (class_name,)
        )
        row = cur.fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()