import socket

ports = {
    "TWS Live": 7496, 
    "TWS Paper": 7497, 
    "IB Gateway Live": 4001, 
    "IB Gateway Paper": 4002
}

print("🔍 Scanning for Interactive Brokers API...")
found = False

for name, port in ports.items():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', port))
    if result == 0:
        print(f"✅ SUCCESS: Found {name} listening on port {port}!")
        found = True
    sock.close()

if not found:
    print("❌ FAILED: No IBKR instances found on standard ports.")
    print("   -> Is Trader Workstation (TWS) open on your Mac?")
    print("   -> Did you enable 'ActiveX and Socket Clients' in TWS API Settings?")
