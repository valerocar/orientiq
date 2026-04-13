/**
 * Google Apps Script backend for OrientIQ landing page lead capture.
 *
 * Setup:
 * 1) Create a Google Sheet with tabs: waitlist, survey, pricing_intent
 * 2) Put headers in row 1:
 *    waitlist: timestamp, email, ref, client_key
 *    survey: timestamp, email, ref, printers, bottleneck, pricing, client_key
 *    pricing_intent: timestamp, email, tier, ref, client_key
 * 3) Set SHEET_ID below
 * 4) Deploy as Web App:
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 5) Copy Web App URL into APPS_SCRIPT_URL in index.html
 */

const SHEET_ID = 'REPLACE_WITH_YOUR_SHEET_ID';
const WAITLIST_TAB = 'waitlist';
const SURVEY_TAB = 'survey';
const PRICING_TAB = 'pricing_intent';
const DEDUPE_TTL_SEC = 90;

function doPost(e) {
  try {
    const payload = parsePayload_(e);
    const eventType = String(payload.event_type || '').trim();
    const email = String(payload.email || '').trim().toLowerCase();
    const hp = String(payload.hp_field || '').trim();
    const clientKey = String(payload.client_key || '').trim();
    const timestamp = String(payload.submitted_at || new Date().toISOString());
    const ref = String(payload.ref || '(direct)');

    // Honeypot: act successful but drop.
    if (hp) return json_({ ok: true, ignored: 'honeypot' });

    if (!eventType) return json_({ ok: false, error: 'Missing event_type' });
    if (!clientKey) return json_({ ok: false, error: 'Missing client_key' });
    if (!isValidEmail_(email)) return json_({ ok: false, error: 'Invalid email' });

    // Lightweight dedupe/rate guard.
    const cache = CacheService.getScriptCache();
    if (cache.get(clientKey)) return json_({ ok: true, deduped: true });
    cache.put(clientKey, '1', DEDUPE_TTL_SEC);

    const sheet = SpreadsheetApp.openById(SHEET_ID);

    if (eventType === 'waitlist') {
      sheet.getSheetByName(WAITLIST_TAB).appendRow([
        timestamp, email, ref, clientKey
      ]);
      return json_({ ok: true });
    }

    if (eventType === 'survey') {
      const answers = payload.answers || {};
      sheet.getSheetByName(SURVEY_TAB).appendRow([
        timestamp,
        email,
        ref,
        String(answers.printers || ''),
        String(answers.bottleneck || ''),
        String(answers.pricing || ''),
        clientKey,
      ]);
      return json_({ ok: true });
    }

    if (eventType === 'pricing_intent') {
      sheet.getSheetByName(PRICING_TAB).appendRow([
        timestamp,
        email,
        String(payload.tier || 'unknown'),
        ref,
        clientKey,
      ]);
      return json_({ ok: true });
    }

    return json_({ ok: false, error: 'Unknown event_type' });
  } catch (err) {
    return json_({ ok: false, error: err.message || 'Server error' });
  }
}

function parsePayload_(e) {
  if (!e || !e.postData || !e.postData.contents) return {};
  return JSON.parse(e.postData.contents);
}

function isValidEmail_(email) {
  if (!email) return false;
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function json_(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
