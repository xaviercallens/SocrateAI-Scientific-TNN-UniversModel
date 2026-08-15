import http.server
import socketserver
import json
import os
import sys

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.dirname(__file__))
from experiment_database import ExperimentDatabase

PORT = 8050
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))

class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/api/ledger':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            db = ExperimentDatabase()
            experiments = db.fetch_all_experiments()
            self.wfile.write(json.dumps(experiments).encode('utf-8'))
        else:
            super().do_GET()

def run_server():
    with socketserver.TCPServer(("", PORT), DashboardRequestHandler) as httpd:
        print(f"==========================================================================")
        print(f" SOCRATE-AI COMMAND CENTER SERVER RUNNING ON: http://localhost:{PORT}")
        print(f" Integrated with SQLite Database Ledger: database/experiment_ledger.db")
        print(f"==========================================================================")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
