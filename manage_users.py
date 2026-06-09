"""User management CLI for the Graph RAG web server.

Usage:
    uv run python manage_users.py add <username> <password>
    uv run python manage_users.py remove <username>
    uv run python manage_users.py list
"""

import argparse
import sys

import bcrypt

from web.user_store import delete_user, init_db, list_users, upsert_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Graph RAG web server users")
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add or update a user's password")
    add_p.add_argument("username")
    add_p.add_argument("password")

    rm_p = sub.add_parser("remove", help="Remove a user and all their sessions")
    rm_p.add_argument("username")

    sub.add_parser("list", help="List all usernames")

    args = parser.parse_args()
    init_db()

    if args.command == "add":
        pw_hash = bcrypt.hashpw(args.password.encode(), bcrypt.gensalt()).decode()
        upsert_user(args.username, pw_hash)
        print(f"User '{args.username}' saved.")
    elif args.command == "remove":
        delete_user(args.username)
        print(f"User '{args.username}' removed.")
    elif args.command == "list":
        users = list_users()
        if users:
            for u in users:
                print(u)
        else:
            print("No users found.", file=sys.stderr)


if __name__ == "__main__":
    main()
