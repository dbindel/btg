"""
  The Error file
"""
module BtgError

export set_msg, set_status, get_status, get_msg

error_status = 0
error_string = ""

function set_msg(s)
    """
    Set the error message [s] to [error_string]
    """
    error_string = s
end

function set_status()
    """
    Set [error_status] to 0
    """
    error_status = 0
end

function get_status()
    """
    Get [error_status]
    """
    return error_status
end

function get_msg()
    """
    Get [error_string]
    """
    return error_string
end

end
