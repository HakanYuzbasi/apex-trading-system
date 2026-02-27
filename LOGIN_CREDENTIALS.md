# Apex Trading System - Login Credentials

## Admin Account

**Username**: `admin`
**Password**: `ApexAdmin!2026`

## Login URL

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## API Authentication

### Get Access Token
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"ApexAdmin!2026"}'
```

### Response
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Use Token in Requests
```bash
curl http://localhost:8000/state \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Password Reset

If you need to change the admin password, update the `.env` file:
```bash
APEX_ADMIN_PASSWORD=YourNewPassword
```

Then restart the API server:
```bash
./apex_ctl.sh restart api
```

## Security Notes

- Access tokens expire after 24 hours
- Refresh tokens can be used to get new access tokens
- Keep these credentials secure - do not commit to version control
- For production deployment, use a strong, unique password

---

**Last Updated**: 2026-02-26
**Environment**: Development/Paper Trading
