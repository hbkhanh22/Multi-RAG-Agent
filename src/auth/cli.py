"""
Account administration, run by the operator on the server (the only way, besides a
verified Google e-mail listed in ADMIN_EMAILS, to get the admin role).

  python -m src.auth.cli admin EMAIL            create an admin, or make an account admin with a NEW password
  python -m src.auth.cli reset-password EMAIL   set a new password, sign the account out everywhere
  python -m src.auth.cli demote EMAIL           back to a normal user
  python -m src.auth.cli claim EMAIL            give conversations made before sign-in existed to EMAIL
  python -m src.auth.cli list

In Docker: docker compose exec api python -m src.auth.cli admin EMAIL
The password is prompted for (or read from stdin when it is not a terminal).
"""

import argparse
import getpass
import sys

from sqlalchemy import select, update

from src.auth.users import AuthError, demote, find_by_email, set_admin, set_password
from src.storage.db import attachments, conversations, generated_images, get_engine, users


def _ask_password() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Repeat password: "):
        raise AuthError("Passwords do not match")
    return first


def claim(user_id: str) -> int:
    with get_engine().begin() as conn:
        for table in (attachments, generated_images):
            conn.execute(update(table).where(table.c.user_id.is_(None)).values(user_id=user_id))
        return conn.execute(update(conversations).where(conversations.c.user_id.is_(None))
                            .values(user_id=user_id)).rowcount


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")       # Vietnamese messages on a Windows console
    parser = argparse.ArgumentParser(prog="python -m src.auth.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("admin", "reset-password", "demote", "claim"):
        sub.add_parser(name).add_argument("email")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "list":
            with get_engine().connect() as conn:
                for r in conn.execute(select(users).order_by(users.c.created_at)):
                    print(f"{r.email}\t{r.role}\t{'active' if r.is_active else 'disabled'}")
        elif args.cmd == "admin":
            user = set_admin(args.email, _ask_password())
            print(f"{user.email} is now an admin (other sessions signed out)")
        else:
            user = find_by_email(args.email)
            if user is None:
                raise AuthError(f"No such account: {args.email}")
            if args.cmd == "reset-password":
                set_password(user.id, _ask_password())
                print("Password changed, all sessions signed out")
            elif args.cmd == "demote":
                demote(user.id)
                print(f"{user.email} is now a normal user")
            else:
                print(f"{claim(user.id)} conversations now belong to {user.email}")
    except AuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
