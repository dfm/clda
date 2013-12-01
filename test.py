from ctr.cf import CF

if __name__ == "__main__":
    import sqlite3
    from multiprocessing import Pool
    with sqlite3.connect("data/abstracts.db") as connection:
        c = connection.cursor()
        c.execute("SELECT user_id,arxiv_id FROM activity")
        activity = c.fetchall()

    model = CF(100)
    pool = Pool()
    model.learn(activity, pool=pool)
