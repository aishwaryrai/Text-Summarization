print("ARTICLE:\n", test_df.iloc[0]['article'])
print("\nGENERATED SUMMARY:\n", summarize(test_df.iloc[0]['article']))
print("\nACTUAL SUMMARY:\n", test_df.iloc[0]['highlights'])
