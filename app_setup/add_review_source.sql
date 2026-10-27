-- One-time upgrade for photo_submissions tables created before review_source existed
-- (create_storage.sql now includes the column). Run once, then deploy the app version that
-- writes review_source; the app's MERGE lists every column, so it needs this column first.
ALTER TABLE dbx_ai_app.airplane_tail_id.photo_submissions
ADD COLUMN review_source STRING COMMENT 'Who set review_label: app_user (feedback in the app) or reviewer (NULL on older reviewed rows means reviewer)';

-- Labels set before this column existed were set by a reviewer. Safe to rerun.
UPDATE dbx_ai_app.airplane_tail_id.photo_submissions
SET review_source = 'reviewer'
WHERE review_label IS NOT NULL AND review_source IS NULL;
