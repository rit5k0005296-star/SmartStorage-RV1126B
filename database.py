# database.py
import sqlite3

def init_db():
    conn = sqlite3.connect('smart_storage.db')
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        role TEXT,
        dorm_info TEXT,
        password TEXT,
        feature BLOB
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        count INTEGER
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        item_id INTEGER,
        borrow_time TIMESTAMP,
        due_time TIMESTAMP, 
        status TEXT, 
        admin_note TEXT DEFAULT 'unprocessed' 
    )''')

    cursor.execute("SELECT COUNT(*) FROM items")
    if cursor.fetchone()[0] == 0:
        new_items = [
            ('剪刀', 10),
            ('胶带', 10),
            ('绷带', 10),
            ('螺丝刀', 10),
            ('胶棒', 10),
            ('直尺', 10),
            ('订书机', 10)
        ]
        cursor.executemany("INSERT INTO items (name, count) VALUES (?,?)", new_items)
        print(f"成功初始化了 {len(new_items)} 种新物品。")
        
    conn.commit()
    conn.close()
    print("数据库初始化完成。")

if __name__ == '__main__':
    init_db()