from ctr.cf import CF

if __name__ == "__main__":
    import sqlite3
    from multiprocessing import Pool
    with sqlite3.connect("data/abstracts.db") as connection:
        c = connection.cursor()
        c.execute("SELECT user_id,arxiv_id FROM activity LIMIT 50000")
        activity = c.fetchall()
        c.execute("""SELECT user_id,count(user_id) FROM activity
                     GROUP BY user_id""")
        users = c.fetchall()

    users = [u for u, count in users if count > 10]
    activity = [(u, a) for u, a in activity if u in users]
    print(len(activity))

    model = CF(100)
    pool = Pool()
    model.learn(activity, pool=pool)
