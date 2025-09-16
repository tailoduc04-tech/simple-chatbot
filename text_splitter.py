from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from config import DOCUMENT_PATH

markdown_path = DOCUMENT_PATH
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
print(len(data))