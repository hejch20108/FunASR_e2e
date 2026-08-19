BEGIN IMMEDIATE;
DELETE FROM recordings WHERE storage_kind = 'legacy_external';
PRAGMA user_version = 2;
COMMIT;
