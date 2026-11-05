# Deploy the AI Interview Agent for free

The app runs on **Streamlit Community Cloud** (free hosting) and uses **Groq's free API**
for the AI interviewer and for speech recognition. Your laptop setup with Ollama keeps
working unchanged: the cloud only uses different settings.

| Part | Service | Free limits (Nov 2026, may change) |
|---|---|---|
| Web app (HTTPS link) | Streamlit Community Cloud | up to ~2.7 GB memory, 2 CPUs; sleeps after 12 h without visitors |
| AI interviewer | Groq, `openai/gpt-oss-20b` | 30 requests/min, 1,000 requests/day, 8,000 tokens/min |
| Speech-to-text | Groq Whisper `whisper-large-v3-turbo` | 20 requests/min, 2,000/day |
| Interviewer voice | edge-tts (gTTS as backup) | no account needed |
| Face detection | Google MediaPipe, runs in the candidate's browser | no account needed |

One interview of 5 questions uses roughly 15 AI requests and 5 speech requests, so the
free daily limits are enough for several dozen interviews a day.

You need two free accounts: **GitHub** and **Groq**. No credit card is required.

---

## Step 1 - Get a free Groq API key (2 minutes)

1. Go to <https://console.groq.com> and sign in (Google or GitHub sign-in works).
2. Open **API Keys** and click **Create API Key**. Name it `interview-app`.
3. Copy the key (it starts with `gsk_`) and keep it somewhere private.
   You cannot see it again later; if you lose it, create a new one.

## Step 2 - Put the code on GitHub (5 minutes)

1. Sign in at <https://github.com> and click **New** (new repository).
2. Name: `ai-voice-interview-agent`. Choose **Public** (simplest for Streamlit Cloud).
   Do not tick "Add a README". Click **Create repository**.
3. On the empty repository page, click **uploading an existing file**.
4. Open the project folder in File Explorer, select everything inside it and drag it into
   the browser. **Before you click Commit, check the list and remove `.env` if it is
   there** (it may contain keys). Also leave out any `venv` folder.
   Make sure the `.streamlit` and `proctor_component` folders were uploaded.
5. Click **Commit changes**.

The repository must contain `app.py` and `requirements.txt` at the top level.

## Step 3 - Deploy on Streamlit Community Cloud (5-10 minutes)

1. Go to <https://share.streamlit.io> and sign in **with GitHub**. Allow access when asked.
2. Click **Create app** -> **Yup, I have an app**.
3. Fill in:
   - **Repository:** `your-github-name/ai-voice-interview-agent`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name, for example `rsb-interview-demo`
4. Click **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste the text below, with your own key and password:

     ```toml
     LLM_PROVIDER = "groq"
     STT_PROVIDER = "groq"
     GROQ_API_KEY = "gsk_your_key_here"
     APP_PASSWORD = "choose-a-password"
     MAX_INTERVIEW_QUESTIONS = "5"
     ```
   Click **Save**.
5. Click **Deploy**. The first start installs the packages and takes a few minutes.

You get a link like `https://rsb-interview-demo.streamlit.app`.

## Step 4 - Test it

1. Open the link in **Chrome or Edge** and enter the password.
2. Run one short interview (3 questions): upload a resume, paste a job description,
   screen it, then go through the camera check and answer by voice.
3. If the camera check says the page is inside another website's frame, click
   **Open the interview in its own tab**. That opens the app's own address, where camera
   and fullscreen are always allowed. Enter the password again and continue.

## Changing settings later

On share.streamlit.io, open your app's menu (**...**) -> **Settings** -> **Secrets**, edit,
and save. The app restarts with the new settings. Any setting from `.env.example` can be
set here the same way (always as `NAME = "value"`).

To update the code, upload the changed files to the GitHub repository again; the app
redeploys automatically.

## Good to know

- **Privacy.** Resumes, job descriptions, answers and voice recordings are sent to Groq
  for processing (Groq states it does not use API data for training). Camera video never
  leaves the browser. For real candidates, get their consent first; for demos, use sample
  resumes. Locally with Ollama and `STT_PROVIDER=faster_whisper`, nothing leaves your laptop.
- **Keep the password on.** The link is public. Without `APP_PASSWORD`, anyone who finds
  it could use up your free Groq quota.
- **Interviews are kept in memory only.** When the app sleeps or restarts, unfinished
  interviews are lost. Download each report at the end (**Download report**).
- **Free limits.** If the daily Groq limit is reached, the app says so; it resets daily.
- **Interviewer voice.** If the free voice service is unreachable from the cloud, the
  question is still shown as text with a "Retry audio" button.

## Troubleshooting

| What you see | What to do |
|---|---|
| "The application could not start: ... GROQ_API_KEY is not set" | Add `GROQ_API_KEY` in Secrets (Step 3.4) and save. |
| "Some project files are from an older version" | A file was not uploaded or is old. Upload all project files again. |
| "The interview AI rejected the credentials" | The Groq key is wrong or was deleted. Create a new one and update Secrets. |
| "The free AI quota for today has been used up" | Wait until the next day, or create the app under another Groq account. |
| App shows "Oh no" / crashed | On share.streamlit.io open **Manage app** (bottom right of the app) to see the logs, then **Reboot app**. |
| Camera or fullscreen blocked | Click **Open the interview in its own tab** on the camera check. Use Chrome or Edge. |
