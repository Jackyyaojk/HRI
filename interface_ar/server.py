import http.server
import socketserver


def main():
    serverPort = 8010
    Handler = http.server.SimpleHTTPRequestHandler
    webServer = socketserver.TCPServer(("", serverPort), Handler)
    print("Server started at port %s" % (serverPort))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.shutdown()
    print("Server stopped.")






if __name__ == "__main__":
    main()