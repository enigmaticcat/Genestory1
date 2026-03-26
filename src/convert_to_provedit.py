import csv
import argparse

def convert_to_provedit(input_csv, output_csv):
    """
    Converts a raw company CSV format to PROVEDIt style CSV format.
    """
    print(f"Reading input file: {input_csv}")
    
    with open(input_csv, 'r', encoding='utf-8') as fin, open(output_csv, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        
        # Prepare output headers
        out_headers = ['Sample File', 'Marker', 'Dye']
        for i in range(1, 101):
            out_headers.extend([f'Allele {i}', f'Size {i}', f'Height {i}'])
            
        writer = csv.DictWriter(fout, fieldnames=out_headers)
        writer.writeheader()
        
        row_count = 0
        for row in reader:
            # If the row is empty or essentially malformed, skip
            if not row.get('Sample File'):
                continue
                
            sample_file = row['Sample File']
            marker = row['Marker']
            dye = row['Dye']
            
            converted_row = {
                'Sample File': sample_file,
                'Marker': marker,
                'Dye': dye
            }
            
            # Extract valid alleles, sizes, heights (up to 4 in current format)
            alleles = []
            sizes = []
            heights = []
            
            for i in range(1, 5):
                allele_val = row.get(f'Allele {i}', '').strip()
                size_val = row.get(f'Size {i}', '').strip()
                height_val = row.get(f'Height {i}', '').strip()
                
                if allele_val:  # not empty
                    alleles.append(allele_val)
                    sizes.append(size_val)
                    heights.append(height_val)
                    
            # Fill in the 100 slots required by PROVEDIt
            for i in range(1, 101):
                idx_0based = i - 1
                if idx_0based < len(alleles):
                    converted_row[f'Allele {i}'] = alleles[idx_0based]
                    converted_row[f'Size {i}'] = sizes[idx_0based]
                    converted_row[f'Height {i}'] = heights[idx_0based]
                else:
                    converted_row[f'Allele {i}'] = ''
                    converted_row[f'Size {i}'] = ''
                    converted_row[f'Height {i}'] = ''
                    
            writer.writerow(converted_row)
            row_count += 1
            
    print(f"Processed {row_count} rows.")
    print(f"Saved output to: {output_csv}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw CSV to PROVEDIt format")
    parser.add_argument("--input", default="data/processed/20251011_HID360.csv", help="Input CSV path")
    parser.add_argument("--output", default="data/processed/20251011_HID360_PROVEDIt.csv", help="Output CSV path")
    args = parser.parse_args()
    
    convert_to_provedit(args.input, args.output)
