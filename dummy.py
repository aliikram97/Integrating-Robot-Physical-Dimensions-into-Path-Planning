import multiprocessing

def process_item(item):
    # Do some processing on the item
    member,flag = item
    if flag:
        result = member * 2
    else:
        result = member
    return result

if __name__ == "__main__":
    # Sample list of items
    my_list = [(1,True), (2,True), (3,False), (4,True), (5,True), (6,False), (7,True), (8,True), (9,False), (10,True)]

    # Number of worker processes
    num_processes = multiprocessing.cpu_count()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the processing of list items among the workers
        results = pool.map(process_item, my_list)

    # Print the results
    print(results)
