package xuhogan.haojames;

public class DimensionMismatchException extends Exception{
	private static final long serialVersionUID = -7531937569150773540L;
	
	public DimensionMismatchException()
	{
	}
	
	public DimensionMismatchException(String message) {
		super(message);
	}
	
	public DimensionMismatchException(Throwable cause) {
		super(cause);
	}
	
	public DimensionMismatchException(String message, Throwable cause) {
		super(message, cause);
	}
	
	public DimensionMismatchException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}
}
