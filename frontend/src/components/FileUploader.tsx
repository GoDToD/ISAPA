interface Props {
    onUpload: (file: File) => void;
  }
  
  export default function FileUploader({ onUpload }: Props) {
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        onUpload(e.target.files[0]);
      }
    };
  
    return (
      <div style={{ marginTop: "10px" }}>
        <input type="file" accept=".pdf,.docx" onChange={handleFileChange} />
      </div>
    );
  }
  