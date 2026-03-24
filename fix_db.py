import re

with open('database.py', 'r') as f:
    content = f.read()

new_method = '''    def get_mysql_connection(self):
        """Get MySQL connection directly."""
        if not self.mysql_pool:
            raise RuntimeError("MySQL is not available.")
        return self.mysql_pool.get_connection()

'''

content = re.sub(
    r'    @contextmanager\s+def get_mysql_connection\(self\):.*?(?=    def |\Z)',
    new_method,
    content,
    flags=re.DOTALL
)

with open('database.py', 'w') as f:
    f.write(content)

print('Done! contextmanager removed.')