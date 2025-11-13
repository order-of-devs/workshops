from packages.create_rag.src.create_rag import load_dataset, initial_rag, add_dataset_to_rag, load_llm, invoke_response

dataset = load_dataset()

rag = initial_rag()
add_dataset_to_rag(rag, dataset)

llm = load_llm()
print(invoke_response(llm, rag, "You are software engineer. Generate an php code  which should get dataframe using flowphp library which should work for example with 3 columns: Name, Age, City and 3 rows. Result should be code in php format only."))

