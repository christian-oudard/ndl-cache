# ndl-cache

A local DuckDB cache in front of the Nasdaq Data Link datatables API.

## Versioning

Bump the patch version in `pyproject.toml` in the same commit as any change to
the package. Every commit that changes behaviour, not once per release.

This is what makes an update reach a machine at all. Consumers install straight
from git, and pip reinstalls a git requirement only when the version it builds
differs from the version already installed. While the version sat at `0.1.0`,
`pip install` and therefore `conda env update` reported the package as already
satisfied and kept running the old code, whatever revision the requirement
named.

That was not hypothetical. A Windows machine ran a January revision until
2026-08-14: Sharadar had dropped the `famasector` column from
`SHARADAR/TICKERS`, the fix had been on this branch since 2026-08-06, and every
update the machine ran skipped it. A second machine was still on an old
revision three days later for the same reason.

A bump on a branch that is not yet merged still counts. If two branches land in
sequence and only the first bumped, anyone who installed in between never gets
the second.

## Cache compatibility

A cached table's layout is part of the interface with whatever revision wrote
it. When a table definition changes, both directions have to keep working: a
newer build must rebuild a layout an older build wrote, and an older build must
still be able to write the rebuilt table. Decide readability before age, a
stale-layout check gated behind a freshness stamp never runs on the machine
that needs it.
