"""User management CLI for the Graph RAG web server.

Usage:
    uv run graph-rag-users add <username>                  # prompts for password
    uv run graph-rag-users add <username> --password-stdin # reads password from stdin
    uv run graph-rag-users remove <username>
    uv run graph-rag-users list
"""

import argparse
import getpass
import sys

import bcrypt

from local_graph_rag.web.security import BCRYPT_MAX_PASSWORD_BYTES, password_fits_bcrypt
from local_graph_rag.web.user_store import delete_user, init_db, list_users, upsert_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Graph RAG web server users")
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add or update a user's password")
    add_p.add_argument("username")
    add_p.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read password from stdin instead of prompting (for scripting)",
    )

    rm_p = sub.add_parser("remove", help="Remove a user and all their sessions")
    rm_p.add_argument("username")

    sub.add_parser("list", help="List all usernames")

    args = parser.parse_args()
    init_db()

    if args.command == "add":
        if args.password_stdin:
            password = sys.stdin.readline().rstrip("\r\n")
        else:
            password = getpass.getpass("Password: ")
        if not password:
            print("Error: password cannot be empty.", file=sys.stderr)
            sys.exit(1)
        if not password_fits_bcrypt(password):
            print(
                f"Error: password cannot exceed {BCRYPT_MAX_PASSWORD_BYTES} bytes.",
                file=sys.stderr,
            )
            sys.exit(1)
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
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
