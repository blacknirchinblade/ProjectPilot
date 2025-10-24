"""
A simple singleton class to share state between different agents or modules.
Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

class SharedMemory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedMemory, cls).__new__(cls)
            # Initialize the data store
            cls._instance.data = {}
        return cls._instance

    def set(self, key, value):
        """Sets a value for a given key in the shared memory."""
        self.data[key] = value

    def get(self, key, default=None):
        """Gets a value for a given key from the shared memory."""
        return self.data.get(key, default)

    def delete(self, key):
        """Deletes a key-value pair from the shared memory."""
        if key in self.data:
            del self.data[key]

    def clear(self):
        """Clears all data from the shared memory."""
        self.data.clear()

    def __str__(self):
        return str(self.data)
