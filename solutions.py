nums = [4, 3, 2, 0, 6, 7, 5,1,1,1,1]

def find_missing_number(nums):
    x = []
    nums =  sorted(nums)
    if nums == []:
        return(0)
    
    for i in range(len(nums)):
        if nums[i] == 0:
            x.append(nums[i])
    return(x)
    
print(find_missing_number(nums))


def longest_consecutive_subsequence(nums):
    x = []
    #nums = sorted(nums)
    if nums == []:
        print("The list is empty")
    if len(nums) >= 1:
        i = 0
        #end = len(nums)
        while i < len(nums) - 1:
            print(i)
            v1 = nums[i]
            v2 = nums[i+1]
            print(v1,v2)
            if v1 + 1 == v2:
                x.append(v1)

            i += 1
            # #if i > 0 and i == len(nums):
            #  #   break
            # else:
            #     i += 1
            #     print(x)

    return(x)


print(longest_consecutive_subsequence([10,25,1,2,3,5,6]))