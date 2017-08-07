#!/usr/bin/env python
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
from config import Config
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread


class db(object):
    def __init__(self, config):
        self.status_message = False
        self.db_schema_file = 'db/db_schema.txt'
        # Pass config -> this class
        for k, v in Config.items():
            setattr(self, k, v)

    def __enter__(self):
        if self.db_ssh_forward:
            forward = sshtunnel.SSHTunnelForwarder(
                credentials.machine_credentials()['ssh_address'],
                ssh_username=credentials.machine_credentials()['username'],
                ssh_password=credentials.machine_credentials()['password'],
                remote_bind_address=('127.0.0.1', 5432))
            forward.start()
            pgsql_port = forward.local_bind_port
            pgsql_string = credentials.postgresql_connection(str(pgsql_port))
            self.forward = forward
        else:
            self.forward = None
        self.pgsql_port = pgsql_port
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        if self.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def recreate_db(self, run=False):
        if run:
            db_schema = open(self.db_schema_file).read().splitlines()
            for s in db_schema:
                t = s.strip()
                if len(t):
                    self.cur.execute(t)
            # self.cur.execute(open(self.db_schema_file).read())
            # self.conn.commit()

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                    )

    def populate_db(self, namedict):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        parent_experiment: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO experiments
            (model_struct, loss_function, wd_type, wd_penalty, optimizer, lr, dataset)
            VALUES
            (%(model_struct)s, %(loss_function)s, %(wd_type)s, %(wd_penalty)s, %(optimizer)s, %(lr)s, %(dataset)s)
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def get_parameters(self, random=True):
        if random:
            self.cur.execute(
                """
                SELECT * from experiments h
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM in_process i
                    WHERE h._id = i.experiment_id
                    )
                ORDER BY random()
                """
            )
        else:
            self.cur.execute(
                """
                SELECT * from experiments h
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM in_process i
                    WHERE h._id = i.experiment_id
                    )
                """
            )
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def update_in_process(self, experiment_id):
        self.cur.execute(
            """
             INSERT INTO in_process
             VALUES
             (%(experiment_id)s)
            """,
            {'experiment_id': experiment_id}
        )
        if self.status_message:
            self.return_status('INSERT')

    def reset_in_process(self):
        self.cur.execute(
            """
            DELETE FROM in_process
            """
        )
        if self.status_message:
            self.return_status('DELETE')

    def update_performance(self, namedict):
        self.cur.execute(
            """
            INSERT INTO performance
            (experiment_id, summary_dir, ckpt_file, training_loss, validation_loss, time_elapsed, training_step)
            VALUES (%(experiment_id)s, %(summary_dir)s, %(ckpt_file)s, %(training_loss)s, %(validation_loss)s, %(time_elapsed)s, %(training_step)s)
            RETURNING _id""",
            namedict
            )
        if self.status_message:
            self.return_status('SELECT')


def get_parameters(random=False):
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        param_dict = db_conn.get_parameters(random=random)
        print param_dict
        if param_dict is not None:
            hp_combo_id = param_dict['_id']
            db_conn.update_in_process(hp_combo_id)
        else:
            hp_combo_id = None
    return param_dict, hp_combo_id


def initialize_database():
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db(run=True)
        db_conn.return_status('CREATE')


def reset_in_process():
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_in_process()
    print 'Cleared the in_process table.'


def main(
        initialize_db,
        reset_process=False):
    if reset_process:
        reset_in_process()
    if initialize_db:
        print 'Initializing database.'
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_process",
        dest="reset_process",
        action='store_true',
        help='Reset the in_process table.')
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))

