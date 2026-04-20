Quick deploy notes (single EC2 instance)

Files created by this repo for deployment:
- deploy/nginx.conf  -> Nginx site config (copy to /etc/nginx/sites-available/caption-app)
- deploy/fastapi.service -> systemd unit for Uvicorn (edit User/WorkingDirectory/PATH)
- backend/.env.example -> backend env template

Important: ensure the model artifact files exist in `backend/model/`:
- InceptionV3_best_model.keras
- tokenizer.pkl
- max_length.pkl

If `tokenizer.pkl` and `max_length.pkl` are not present, copy them from your training outputs into `backend/model/` before starting the service.

